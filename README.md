# TorchImage

TorchImage is a .NET library that seamlessly integrates powerful image processing capabilities into TorchSharp using SixLabors.ImageSharp. It provides datasets and utilities for loading and transforming images, making it easier to work with image data in machine learning and computer vision projects.

## Features

- Load images from folders and apply transformations.
- Support for various image formats including PNG, JPG, JPEG, BMP, GIF, TIFF, and WEBP.
- Integration with TorchSharp for tensor operations.
- Easy-to-use dataset classes for image loading and processing.

## Installation

To install TorchImage, add the following package references to your .NET project:
```xml
<PackageReference Include="TorchImage" Version="0.0.1" />
```
## Usage
- ### ImageFolderDataset

	The `ImageFolderDataset` class allows you to load images from a folder and apply transformations.
- ### SimpleImageDataset

	The `SimpleImageDataset` class allows you to load images from a specified root path and apply transformations.
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
