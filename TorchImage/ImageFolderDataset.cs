using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace TorchImage;
/// <summary>
/// Dataset for loading images from a folder.
/// 
/// Extends <see cref="torch.utils.data.Dataset"/> and implements the <see cref="torch.utils.data.Dataset.GetTensor"/> method.
/// </summary>
public class ImageFolderDataset : torch.utils.data.Dataset
{
    private readonly (string path, int label)[] _samples;
    private readonly ITransform? _transform;

    /// <summary>
    /// Gets the classes in the dataset.
    /// </summary>
    public string[] Classes { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ImageFolderDataset"/> class.
    /// </summary>
    /// <param name="rootDir">The root directory containing the image folders.</param>
    /// <param name="transform">The transform to apply to the images.</param>
    /// <param name="searchPattern">The search pattern for image files.</param>
    /// <exception cref="DirectoryNotFoundException">Thrown when the root directory is not found.</exception>
    public ImageFolderDataset(
        string rootDir,
        ITransform? transform = null,
        string searchPattern = "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff;*.webp") : base()
    {
        // Check if the root directory exists
        if (!Directory.Exists(rootDir))
            throw new DirectoryNotFoundException($"Directory '{rootDir}' not found.");

        Classes = [.. Directory.GetDirectories(rootDir)
            .Select(Path.GetFileName)
            .OrderBy(c => c)];

        var classToIdx = Classes.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);
        _samples = Directory.GetDirectories(rootDir)
            .SelectMany(dir => Directory.GetFiles(dir, searchPattern)
            .Select(file => (file, classToIdx[Path.GetFileName(Path.GetDirectoryName(file)!)]))
            .ToArray()).Select(x => (x.file, x.Item2)).ToArray();

        _transform = transform;
    }

    /// <summary>
    /// Gets the number of samples in the dataset.
    /// </summary>
    public override long Count => _samples.Length;

    /// <summary>
    /// Gets the tensor representation of the image at the specified index.
    /// 
    /// Shape: CxHxW
    /// </summary>
    /// <param name="index">The index of the image.</param>
    /// <returns>A dictionary containing the image tensor and its label.</returns>
    public override Dictionary<string, Tensor> GetTensor(long index)
    {
        var (path, label) = _samples[index];
        var image = Image.Load<Rgb24>(path);
        var tensor = TorchImageUtils.ImageToTensor(image);
        if (_transform != null)
            tensor = _transform.call(tensor);
        return new Dictionary<string, Tensor> { { "image", tensor }, { "label", torch.tensor(label) } };
    }

    /// <summary>
    /// Converts class index to class name.
    /// </summary>
    /// <param name="index">The index of the class.</param>
    /// <returns>The name of the class.</returns>
    public string GetClassName(int index)
    {
        return Classes[index];
    }

    /// <summary>
    /// Converts class name to class index.
    /// </summary>
    /// <param name="className">The name of the class.</param>
    /// <returns>The index of the class.</returns>
    public int GetClassIndex(string className)
    {
        return Array.IndexOf(Classes, className);
    }
}
