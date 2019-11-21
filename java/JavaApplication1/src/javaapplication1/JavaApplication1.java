/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javaapplication1;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
/**
 *
 * @author papa
 */
public class JavaApplication1 {
    float[] IDENTITY = {0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f};
    BufferedImage normal;
    BufferedImage current;
    Image image;
    /**
     * @param args the command line arguments
     */
    public  void apply() {
        // TODO code application logic here
        try {
            File pathToFile = new File("image.png");
            image = ImageIO.read(pathToFile);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        int w=image.getWidth(null);
        int h=image.getHeight(null);
        normal=new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Kernel kernel =new Kernel(3, 3, IDENTITY);
        ConvolveOp co=new ConvolveOp(kernel);
        current=co.filter(normal, null);
    }
}
