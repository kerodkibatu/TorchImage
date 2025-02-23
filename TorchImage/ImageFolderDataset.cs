using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace TorchImage;

public class ImageFolderDataset : torch.utils.data.Dataset
{
    private readonly (string path, int label)[] _samples;
    private readonly ITransform? _transform;

    public string[] Classes { get; }

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

    public override long Count => _samples.Length;

    public override Dictionary<string,Tensor> GetTensor(long index)
    {
        var (path, label) = _samples[index];
        var image = Image.Load<Rgb24>(path);
        var tensor = TorchImageUtils.ImageToTensor(image);
        if (_transform != null)
            tensor = _transform.call(tensor);
        return new Dictionary<string, Tensor> { { "image", tensor }, { "label", torch.tensor(label) } };
    }
}
