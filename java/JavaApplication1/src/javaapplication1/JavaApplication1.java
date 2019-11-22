package javaapplication1;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
public class JavaApplication1 {
    int w=0;// ширина входного изображения
    int h=0;// его высота
    float[] imageInputR=new float [w*h];
    float[] imageInputG=new float [w*h];
    float[] imageInputB=new float [w*h];
    /*
    Произвольные веса
    */
    float[][] kernel = new float[][]{{1.0f, 0.8f, 0.6f}, {0.1f, 0.5f, 0.4f}, {0.3f, 0.5f, 0.7f}};
    /*
    Сворачивать по отдельности в отдельную карту признаков,
    т.к. сложно представить сворачивание с 3D ядром.
    */
    float[] featuredMapR;
    float[] featuredMapG;
    float[] featuredMapB;
    public  void work() {
        byte[] imageInputRByte = getByteArray(open("test.png", "R"), "png");
        for (int i = 0; i < imageInputRByte.length; i++) {
            byte b = imageInputRByte[i];
            imageInputR[i]=(float)(b/255.0f);
        }
        featuredMapR=conv(imageInputR, w, h, kernel,1, -1);
    }
    public BufferedImage open(String fName, String whatChannel) {
        Image image = null;
        try {
            File pathToFile = new File(fName);
            image = ImageIO.read(pathToFile);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        w = image.getWidth(null);
        h = image.getHeight(null);
        BufferedImage C = new BufferedImage(
                w, h,
                BufferedImage.TYPE_INT_RGB
        );
        int rgb = 0;
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                rgb = C.getRGB(x, y);
                switch (whatChannel) {
                    case "R":
                        C.setRGB(x, y, rgb & 0xff0000);
                    case "G":
                        C.setRGB(x, y, rgb & 0xff00);
                    case "B":
                        C.setRGB(x, y, rgb & 0xff);
                    default:
                        throw new AssertionError();
                }
            }
        }
        return C;
    }
    public byte[] getByteArray(BufferedImage image, String ext) {
        ByteArrayOutputStream out = null;
        try {
            out = new ByteArrayOutputStream();
            ImageIO.write(image, ext, out);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return out.toByteArray();
    }
    /**
     * Сворачивает 'один' канал(т.к. kernel 2d)
     *
     * @param input изображение как лента
     * @param width ширина изначального изображения
     * @param height его высота
     * @param kernel 2d ядро
     * @param stride шаг при свертке
     * @param isConv 1 для конвульции и -1 для кросс-кореляции
     * @return выходная карта признаков как лента
     */
    public float[] conv(float[] input, int width, int height, float[][] kernel, int stride, int isConv) {
        int kernelWidth = kernel[0].length;
        int kernelHeight = kernel.length;
        int V = (width - kernelWidth / stride) + 1;// выходной обьем(ширина) выходной карты признаков
        float output[] = new float[V * V]; // выходная карта признаков квадратная
        int n = -1;
        //Производим вычисления
        for (int i = 0; i < height - 1; i++) {
            for (int j = 0; j < width - 1; j++) {
                float result = 0;
                n += stride;
                for (int a = 0; a < kernelHeight; a++) {
                    for (int b = 0; b < kernelWidth; b++) {
                        int Y = i * stride + a;
                        int X = j * stride + b;
                        // для прямого распространения - кросс-кореляция,для обратного-ковулция
                        result += input[width * Y - isConv * X] * kernel[a][b];
                        output[n] = result;
                    }
                }
            }
        }
        return output;
    }
    public static void main(String[] args) {
        JavaApplication1 app=new JavaApplication1();
        app.work();
    }
}
