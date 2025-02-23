using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace TorchImage;

public class SimpleImageDataset : torch.utils.data.Dataset
{
    private readonly string[] _samples;
    private readonly ITransform? _transform;

    public SimpleImageDataset(string rootPath, ITransform? transform = null) : base()
    {
        var files = Directory.GetFiles(rootPath, "*.png", SearchOption.AllDirectories);
        _samples = files;
        _transform = transform;
    }

    public override long Count => _samples.Length;

    public override Dictionary<string, Tensor> GetTensor(long index)
    {
        if (index < 0 || index >= _samples.Length)
            throw new ArgumentOutOfRangeException(nameof(index));
        var path = _samples[index];
        var image = Image.Load<Rgb24>(path);
        var tensor = TorchImageUtils.ImageToTensor(image);
        if (_transform != null)
            tensor = _transform.call(tensor);
        return new Dictionary<string, Tensor> { { "image", tensor } };
    }
}
