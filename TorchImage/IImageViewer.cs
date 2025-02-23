using SixLabors.ImageSharp;

namespace TorchImage.ImageViewer;

// Interface for a Preview Surface that can display an image
public interface IImageViewer
{
    void Show(Image image);
}
