using SixLabors.ImageSharp;
using System.Drawing.Drawing2D;
namespace TorchImage.ImageViewer.WinForms;
public class WinFormsViewer : IImageViewer
{
    public static WinFormsViewer Default { get; } = new WinFormsViewer();
    public WinFormsViewer((int width, int height)? minSize = null)
    {
        MinimumSize = minSize.HasValue ? new System.Drawing.Size(minSize.Value.width, minSize.Value.height) : new System.Drawing.Size(512, 512);
    }

    System.Drawing.Size MinimumSize { get; }

    public void Show(SixLabors.ImageSharp.Image image)
    {
        var form = new ImageForm(image)
        {
            MinimumSize = MinimumSize
        };
        form.ShowDialog();
    }
    public void Show(IEnumerable<SixLabors.ImageSharp.Image> images)
    {
        var form = new ImageSlideForm(images)
        {
            MinimumSize = MinimumSize
        };
        form.ShowDialog();
    }
}
class ImageSlideForm : Form
{
    private PictureBox pictureBox;
    private Button nextButton;
    private Button previousButton;
    private NumericUpDown numericUpDown;

    private int index = 0;
    private readonly IEnumerable<System.Drawing.Image> _images;
    public ImageSlideForm(IEnumerable<SixLabors.ImageSharp.Image> images)
    {
        this._images = images.Select(image =>
        {
            var memoryStream = new System.IO.MemoryStream();
            image.SaveAsJpeg(memoryStream);
            var img = System.Drawing.Image.FromStream(memoryStream);
            memoryStream.Dispose();
            return img;
        });
        pictureBox = new PictureBoxWithInterpolationMode(InterpolationMode.NearestNeighbor)
        {
            Dock = DockStyle.Fill,
            SizeMode = PictureBoxSizeMode.Zoom
        };

        pictureBox.Paint += (sender, e) =>
        {
            e.Graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
        };

        nextButton = new Button
        {
            Text = "Next",
            Dock = DockStyle.Right,
        };
        previousButton = new Button
        {
            Text = "Previous",
            Dock = DockStyle.Left
        };

        numericUpDown = new NumericUpDown
        {
            Dock = DockStyle.Bottom,
            Minimum = 1,
            Maximum = _images.Count(),
            Value = 1,
            UpDownAlign = LeftRightAlignment.Right,
            TextAlign = HorizontalAlignment.Center
        };

        numericUpDown.ValueChanged += (sender, e) =>
        {
            index = (int)numericUpDown.Value - 1;
            ShowImage();
        };

        nextButton.Size = previousButton.Size = new System.Drawing.Size(100, 50);

        nextButton.Click += NextButton_Click!;
        previousButton.Click += PreviousButton_Click!;
        Controls.Add(pictureBox);
        Controls.Add(nextButton);
        Controls.Add(previousButton);
        Controls.Add(numericUpDown);
        ShowImage();
        ClientSize = _images.First().Size + new System.Drawing.Size(200, 0);
        // When Resize event is triggered, resize the image to fit the window
        Resize += (sender, e) =>
        {
            pictureBox.Size = ClientSize - new System.Drawing.Size(200, 0);
        };
    }

    private void PreviousButton_Click(object sender, EventArgs e)
    {
        numericUpDown.Value = numericUpDown.Value == 1 ? _images.Count() : numericUpDown.Value - 1;
    }

    private void NextButton_Click(object sender, EventArgs e)
    {
        numericUpDown.Value = numericUpDown.Value == _images.Count() ? 1 : numericUpDown.Value + 1;
    }

    private void ShowImage()
    {
        Text = $"Image Slide Show | {index + 1} of {_images.Count()}";
        pictureBox.Image?.Dispose();
        pictureBox.Image = _images.Skip(index).First();
    }
}
class ImageForm : Form
{
    private PictureBox pictureBox;
    public ImageForm(SixLabors.ImageSharp.Image image)
    {
        pictureBox = new PictureBoxWithInterpolationMode(InterpolationMode.NearestNeighbor)
        {
            Dock = DockStyle.Fill,
            SizeMode = PictureBoxSizeMode.Zoom
        };
        var memoryStream = new System.IO.MemoryStream();
        image.SaveAsPng(memoryStream);

        pictureBox.Image = System.Drawing.Image.FromStream(memoryStream);

        Controls.Add(pictureBox);
        Text = "Image Preview";
        ClientSize = pictureBox.Image.Size;

        memoryStream.Dispose();
    }
}
internal class PictureBoxWithInterpolationMode : PictureBox
{
    InterpolationMode _InterpolationMode { get; set; }
    public PictureBoxWithInterpolationMode(InterpolationMode interpolationMode) : base()
    {
        _InterpolationMode = interpolationMode;
    }

    protected override void OnPaint(PaintEventArgs paintEventArgs)
    {
        paintEventArgs.Graphics.InterpolationMode = _InterpolationMode;
        paintEventArgs.Graphics.PixelOffsetMode = PixelOffsetMode.Half;
        base.OnPaint(paintEventArgs);
    }
}
