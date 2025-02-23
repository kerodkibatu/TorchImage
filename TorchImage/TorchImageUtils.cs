using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace TorchImage;

public static class TorchImageUtils
{
    public static ImageFolderDataset ImageFolderDataset(string rootDir, ITransform transform = null)
    {
        return new ImageFolderDataset(rootDir, transform);
    }
    public static Tensor ImageToTensor(Image<Rgb24> image)
    {
        int height = image.Height;
        int width = image.Width;
        byte[] bytes = new byte[height * width * 3];
        image.CopyPixelDataTo(bytes);

        var tensor = torch.tensor(bytes, torch.uint8)
            .reshape(height, width, 3)
            .permute(2, 0, 1)  // Convert to CxHxW
            .to_type(torch.float32)
            .div(255f);  // Normalize to [0, 1]

        return tensor;
    }

    public static Image<Rgb24> TensorToImage(Tensor tensor)
    {
        tensor = tensor.to_type(torch.float32).detach().cpu();

        if (tensor.dim() != 3)
            throw new ArgumentException("Tensor must be 3D (CxHxW)");

        tensor = tensor.permute(1, 2, 0);  // Convert to HxWxC
        tensor = (torch.clamp(tensor, 0, 1) * 255).to_type(torch.uint8);
        int height = (int)tensor.shape[0];
        int width = (int)tensor.shape[1];
        int channels = (int)tensor.shape[2];

        var image = new Image<Rgb24>(width, height);

        if (channels == 1)
        {
            var memoryStream = new System.IO.MemoryStream();
            tensor.contiguous().WriteBytesToStream(memoryStream);
            var bytes = memoryStream.ToArray();
            image.DangerousTryGetSinglePixelMemory(out var pixelMemory);
            var pixelSpan = pixelMemory.Span;
            for (int i = 0; i < bytes.Length; i++)
            {
                pixelSpan[i] = new Rgb24(bytes[i], bytes[i], bytes[i]);
            }
        }
        else if (channels == 3)
        {
            var memoryStream = new System.IO.MemoryStream();
            tensor.contiguous().WriteBytesToStream(memoryStream);
            var bytes = memoryStream.ToArray();
            image.DangerousTryGetSinglePixelMemory(out var pixelMemory);
            var pixelSpan = pixelMemory.Span;
            for (int i = 0; i < bytes.Length; i += 3)
            {
                pixelSpan[i / 3] = new Rgb24(bytes[i], bytes[i + 1], bytes[i + 2]);
            }
        }
        else
        {
            throw new ArgumentException("Tensor must have 1 or 3 channels");
        }

        

        return image;
    }

    public static Image<Rgba32> TensorToImageAlpha(Tensor tensor)
    {
        tensor = tensor.to_type(torch.float32).detach().cpu();
        if (tensor.dim() != 3)
            throw new ArgumentException("Tensor must be 3D (CxHxW)");
        tensor = tensor.permute(1, 2, 0);  // Convert to HxWxC
        int height = (int)tensor.shape[0];
        int width = (int)tensor.shape[1];

        if (tensor.shape[2] == 4)
        {
            var image = new Image<Rgba32>(width, height);
            var memoryStream = new System.IO.MemoryStream();
            tensor.contiguous().WriteBytesToStream(memoryStream);
            var bytes = memoryStream.ToArray();
            image.DangerousTryGetSinglePixelMemory(out var pixelMemory);
            var pixelSpan = pixelMemory.Span;
            for (int i = 0; i < bytes.Length; i += 4)
            {
                pixelSpan[i / 4] = new Rgba32(bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]);
            }
            return image;
        }
        else
        {
            throw new ArgumentException("Tensor must have 4 channels");
        }
    }

    public static Tensor ImageAlphaToTensor(Image<Rgba32> image)
    {
        int height = image.Height;
        int width = image.Width;
        byte[] bytes = new byte[height * width * 4];
        image.CopyPixelDataTo(bytes);
        var tensor = torch.tensor(bytes, torch.uint8)
            .reshape(height, width, 4)
            .permute(2, 0, 1)  // Convert to CxHxW
            .to_type(torch.float32)
            .div(255f);  // Normalize to [0, 1]

        return tensor;
    }

    public static Tensor ImageArrayToTensor(IEnumerable<Image<Rgb24>> images)
    {
        var tensors = images.Select(ImageToTensor).ToArray();
        return torch.stack(tensors);
    }

    public static IEnumerable<Image<Rgb24>> TensorToImageArray(Tensor tensor)
    {
        Image<Rgb24>[] images = new Image<Rgb24>[tensor.shape[0]];
        for (int i = 0; i < tensor.shape[0]; i++)
        {
            var image = TensorToImage(tensor[i]);
            images[i] = image;
        }
        return images;
    }

    public static Image<Rgb24> MakeGrid(Tensor tensor, int nrow = 8, int padding = 2)
    {
        var grid = torchvision.utils.make_grid(tensor, nrow: nrow, padding: padding);
        return TensorToImage(grid);
    }
    public static Image<Rgb24> MakeGrid(IEnumerable<Tensor> tensors, int nrow = 8, int padding = 2)
    {
        var tensor = torch.stack(tensors);
        return MakeGrid(tensor, nrow, padding);
    }

}
