from google.cloud import vision
import argparse
import cv2

def read_character(img, context):
    key_path = r"goemon-277006-7e564db6fd09.json"
    client = vision.ImageAnnotatorClient.from_service_account_json(key_path)
    image_byte_array = cv2.imencode('.jpg', img)[1].tobytes()
    image = vision.Image(content=image_byte_array)
    response = client.text_detection(image=image, image_context=context)
    symbol_coors = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        text = symbol.text
                        coor_min = symbol.bounding_box.vertices[0]
                        coor_max = symbol.bounding_box.vertices[2]
                        # symbol_coors.append([text, coor_min.x, coor_min.y, coor_max.x, coor_max.y])
                        symbol_coors.append(text)
    result = ''.join(symbol_coors)
    return result

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Read characters from an image")
    parser.add_argument("image_name", type=str, help="Path to the image file")
    args = parser.parse_args()
    # Use GGAPI
    img_path = f"/home/minhtuan/Desktop/VJ/CheckLogBug/img_bug/{args.image_name}"
    image_context_handwrite = vision.ImageContext(language_hints=['ja-t-i0-handwrit'])
    image_context = vision.ImageContext(language_hints=['ja'])
    img = cv2.imread(img_path)
    print("Ket qua su dung viet tay:", read_character(img, image_context_handwrite))
    print("Ket qua su dung danh may:", read_character(img, image_context))