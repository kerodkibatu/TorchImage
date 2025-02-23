using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchImage.ImageViewer;
using static TorchSharp.torch;

namespace TorchImage;

// Tensor extension methods
public static class TensorExtensions
{
    public static void ShowAsImage(this Tensor tensor, IImageViewer preview)
    {
        var image = TorchImageUtils.TensorToImage(tensor);
        preview.Show(image);
    }

    public static Image<Rgb24> ToImage(this Tensor tensor)
    {
        return TorchImageUtils.TensorToImage(tensor);
    }

    public static Image<Rgba32> ToImageAlpha(this Tensor tensor)
    {
        return TorchImageUtils.TensorToImageAlpha(tensor);
    }
}