import cv2

def drawRectangle(image, position, color=(255, 0, 0), thickness=3):
    """
    Draws rectangle on top of given image

    
    Args:
        image (cv2.Image): Input image
        position (startX, startY, endX, endY): coordinates to draw rectangle
        color (B, G, R): Color to draw border
        thickness (int): Border thickness

    Returns:
        image (cv2.Image): Image with drawn rectangle
    """

    # Unpack Positions
    (startX, startY, endX, endY) = position

    return cv2.rectangle(image, (startX, startY), (endX, endY), color, thickness)



def drawString(image, txt, position, color=(255, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2):
    """
    Draws string on top of given image

    
    Args:
        image (cv2.Image): Input image
        txt (str): String to draw
        position (x, y): coordinates to draw rectangle
        color (B, G, R): Color to draw border
        font (cv2.Font) : Font of text
        fontScale (float): scale of font
        thickness (int): Border thickness

    Returns:
        image (cv2.Image): Image with drawn string

    """

    return cv2.putText(image, txt, position, font, fontScale, color, thickness, cv2.LINE_AA)