using SixLabors.ImageSharp;

namespace TorchImage;

/// <summary>
/// Interface for displaying images.
/// Install the TorchImage.WinForms package to use the default windows implementation.
/// </summary>
public interface IImageViewer
{
    void Show(Image image);
}
