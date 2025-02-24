using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchImage;
using static TorchSharp.torch;

namespace TorchImage;

/// <summary>
/// Extension methods for working with tensor images.
/// </summary>
public static class TensorExtensions
{
    /// <summary>
    /// Displays the tensor as an image.
    /// </summary>
    /// <param name="tensor">The tensor to display.</param>
    /// <param name="preview">The image viewer to use.</param>
    public static void ShowAsImage(this Tensor tensor, IImageViewer preview)
    {
        var image = TorchImageUtils.TensorToImage(tensor);
        preview.Show(image);
    }

    /// <summary>
    /// Converts the tensor to an image.
    /// </summary>
    /// <param name="tensor">The tensor to convert.</param>
    public static Image<Rgb24> ToImage(this Tensor tensor)
    {
        return TorchImageUtils.TensorToImage(tensor);
    }
    /// <summary>
    /// Converts the tensor to an image with alpha channel.
    /// </summary>
    /// <param name="tensor">The tensor to convert.</param>
    public static Image<Rgba32> ToImageAlpha(this Tensor tensor)
    {
        return TorchImageUtils.TensorToImageAlpha(tensor);
    }
}