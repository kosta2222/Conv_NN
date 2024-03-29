public static BufferedImage[] splitToRGB(BufferedImage original) {
    BufferedImage R =  new BufferedImage(
            original.getWidth(), original.getHeight(),
            BufferedImage.TYPE_INT_RGB
    );
    BufferedImage G =  new BufferedImage(
            original.getWidth(), original.getHeight(),
            BufferedImage.TYPE_INT_RGB
    );
    BufferedImage B =  new BufferedImage(
            original.getWidth(), original.getHeight(),
            BufferedImage.TYPE_INT_RGB
    );

    for (int x = 0; x < original.getWidth(); x++)
        for (int y = 0; y < original.getHeight(); y++) {
            final int rgb = original.getRGB(x, y);

            R.setRGB(x, y, rgb & 0xff0000);
            G.setRGB(x, y, rgb & 0xff00);
            B.setRGB(x, y, rgb & 0xff);
        }

    return new BufferedImage[]{R, G, B};
}

public static BufferedImage mergeRGB(BufferedImage R, BufferedImage G, BufferedImage B) {
    BufferedImage original =  new BufferedImage(
            R.getWidth(), R.getHeight(),
            BufferedImage.TYPE_INT_RGB
    );

    for (int x = 0; x < original.getWidth(); x++)
        for (int y = 0; y < original.getHeight(); y++) {
            final int rgb = R.getRGB(x, y) | G.getRGB(x, y) | B.getRGB(x, y);
            original.setRGB(x, y, rgb);
        }

    return original;
}  