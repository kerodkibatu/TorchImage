using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace TorchImage;

/// <summary>
/// Represents a simple image dataset that loads images from a specified root path.
/// </summary>
public class SimpleImageDataset : torch.utils.data.Dataset
{
    private readonly string[] _samples;
    private readonly ITransform? _transform;

    /// <summary>
    /// Constructs a new instance of the <see cref="SimpleImageDataset"/> class.
    /// </summary>
    /// <param name="rootPath">The root path to the images.</param>
    /// <param name="transform">The transform to apply to the images.</param>
    public SimpleImageDataset(string rootPath, ITransform? transform = null) : base()
    {
        var supportedExtensions = new[] { "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif" };
        var files = supportedExtensions.SelectMany(ext => Directory.GetFiles(rootPath, ext, SearchOption.AllDirectories)).ToArray();
        _samples = files;
        _transform = transform;
    }

    /// <summary>
    /// The number of samples in the dataset.
    /// </summary>
    public override long Count => _samples.Length;

    /// <summary>
    /// Returns the tensor for the specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
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
