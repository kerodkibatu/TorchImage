using TorchImage;
using TorchImage.ImageViewer.WinForms;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchvision;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
[DllImport("kernel32.dll")]
static extern bool AllocConsole();

AllocConsole();


var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;


var transform = transforms.Compose(
    transforms.Resize(70),
    transforms.CenterCrop(64)
// Combine 3 channels into 1 channel
// transforms.Lambda(img => img.mean([0]).unsqueeze(0))
);

var brainTumorDataset = new SimpleImageDataset("C:\\Users\\Kerod\\Desktop\\TestFolder\\20250216_131429[1]", transform: transform);

var dataLoader = new DataLoader(brainTumorDataset, batchSize: 4, shuffle: true, device: device);

// Sample 16 images from the dataset and display them
var previewImgs = Enumerable.Range(0, 10).Select(i => brainTumorDataset.GetTensor(i)["image"]).Select(TorchImageUtils.TensorToImage).ToList();

WinFormsViewer.Default.Show(previewImgs);


//var AEmodel = new AutoEncoder().to(device);
//AEmodel.Train(dataLoader, numEpochs: 150);
//AEmodel.Test(dataLoader, numTestImages: 100);

var GANmodel = new SimpleGAN(100, 64).to(device);
GANmodel.Train(dataLoader, numEpochs: 150);

class SimpleGAN : nn.Module<Tensor, Tensor>
{
    public Sequential generator;
    public Sequential discriminator;
    public Sequential model;
    public SimpleGAN(long latentDim, long imgSize) : base("SimpleGAN")
    {
        generator = Sequential(
            Linear(latentDim, 128),
            ReLU(),
            Linear(128, 256),
            ReLU(),
            Linear(256, 512),
            ReLU(),
            Linear(512, 1024),
            ReLU(),
            Linear(2048, 4096),
            ReLU(),
            Linear(4096, imgSize * imgSize * 3),
            Sigmoid()
        );
        discriminator = Sequential(
            Linear(imgSize * imgSize * 3, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 1),
            Sigmoid()
        );
        model = Sequential(
            generator,
            discriminator
        );

        RegisterComponents();
    }
    public override Tensor forward(Tensor input)
    {
        return model.forward(input);
    }
    public Tensor Generate(Tensor input)
    {
        return generator.forward(input);
    }
    public Tensor Discriminate(Tensor input)
    {
        return discriminator.forward(input);
    }

    public void Train(DataLoader dataloader, int numEpochs = 100)
    {
        var device = dataloader.Device;

        // Initialize Weights of the model
        foreach (var module in model.children())
        {
            if (module is Linear)
            {
                var linear = (Linear)module;
                nn.init.normal_(linear.weight, 0.0f, 0.02f);
            }
        }

        var criterion = nn.BCELoss();
        var optimizerG = torch.optim.Adam(generator.parameters(), lr: 0.0002f, beta1: 0.5f, beta2: 0.999f);
        var optimizerD = torch.optim.Adam(discriminator.parameters(), lr: 0.0002f, beta1: 0.5f, beta2: 0.999f);
        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            float runningDloss = 0.0f;
            float runningGloss = 0.0f;

            foreach (var batch in dataloader)
            {
                var real = batch["image"];
                var batchSize = real.shape[0];
                var realLabel = torch.ones(batchSize, 1).to(device);
                var fakeLabel = torch.zeros(batchSize, 1).to(device);
                // Train the discriminator
                var output = discriminator.forward(real);
                var lossDReal = criterion.forward(output, realLabel);
                optimizerD.zero_grad();
                lossDReal.backward();
                optimizerD.step();
                var noise = torch.randn(batchSize, 100).to(device);
                var fake = generator.forward(noise);
                output = discriminator.forward(fake.detach());
                var lossDFake = criterion.forward(output, fakeLabel);
                optimizerD.zero_grad();
                lossDFake.backward();
                optimizerD.step();
                // Train the generator
                output = discriminator.forward(fake);
                var lossG = criterion.forward(output, realLabel);
                optimizerG.zero_grad();
                lossG.backward();
                optimizerG.step();

                runningDloss += lossDReal.item<float>() + lossDFake.item<float>();
                runningGloss += lossG.item<float>();
            }

            Console.WriteLine($"Epoch {epoch + 1}/{numEpochs} | D Loss: {runningDloss / dataloader.Count} | G Loss: {runningGloss / dataloader.Count}");
        }
    }
}