

using System.Diagnostics;
using TorchImage;
using TorchImage.ImageViewer.WinForms;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

class AutoEncoder : nn.Module<Tensor, Tensor>
{
    // Simple Autoencoder input size is 1x224x224 
    public Sequential encoder;

    public Sequential decoder;


    public Sequential model;
    public AutoEncoder() : base("AE")
    {
        encoder = Sequential(
            Conv2d(3, 16, 3, padding: 1),
            ReLU(),
            MaxPool2d(2),
            Conv2d(16, 8, 3, padding: 1),
            ReLU(),
            MaxPool2d(2),
            Conv2d(8, 8, 3, padding: 1),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(8 * 28 * 28, 4096)
        );
        decoder = Sequential(
            Linear(4096, 8 * 28 * 28),
            Unflatten(1, [8, 28, 28]),
            ReLU(),
            ConvTranspose2d(8, 8, 3, stride: 2, padding: 1, output_padding: 1),
            ReLU(),
            ConvTranspose2d(8, 16, 3, stride: 2, padding: 1, output_padding: 1),
            ReLU(),
            ConvTranspose2d(16, 3, 3, stride: 2, padding: 1, output_padding: 1),
            Sigmoid()
        );

        model = Sequential(
            encoder,
            decoder
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return model.forward(input);
    }

    public Tensor Encode(Tensor input)
    {
        return encoder.forward(input);
    }

    public Tensor Decode(Tensor input)
    {
        return decoder.forward(input);
    }

    public void Train(DataLoader dataLoader, int numEpochs = 100)
    {
        var optimizer = torch.optim.Adam(model.parameters(), lr: 1e-3f);
        var loss = nn.MSELoss();
        var sw = Stopwatch.StartNew();

        model.train();

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            float runningLoss = 0.0f;
            foreach (var batch in dataLoader)
            {
                optimizer.zero_grad();
                var images = batch["image"];
                var output = model.forward(images);

                var l = loss.forward(output, images);

                l.backward();
                optimizer.step();
                runningLoss += l.item<float>();
            }
            Console.WriteLine($"Epoch {epoch + 1}, loss: {runningLoss / dataLoader.Count}");
        }

        Console.WriteLine($"Training took {sw.ElapsedMilliseconds}ms");

        // Save the model
        model.save("mdl.dat");
        model.eval();
    }

    public void Test(DataLoader dataLoader, int numTestImages = 100)
    {
        var result = new List<Image<Rgb24>>();

        for (int i = 0; i < numTestImages; i++)
        {
            var img = dataLoader.Take(1).First()["image"][0];

            var decoded = model.forward(img.unsqueeze(0))[0];

            var sideBySide = TorchImageUtils.MakeGrid([img, decoded], nrow: 2, padding: 5);
            result.Add(sideBySide);
        }

        WinFormsViewer.Default.Show(result);
    }
}