import cv2
import numpy as np
import imghdr
import logging
import os
from typing import Any, List, Tuple, Optional
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ställer in loggningskonfigurationen
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_advanced_model(model_path: str = None) -> Any:
    """
    Laddar en avancerad modell för objektdetektion eller bildförbättring.
    Om ingen sökväg anges, laddas EfficientNetB0 med förtränade ImageNet-vikter.
    """
    if model_path:
        return load_model(model_path)
    else:
        return EfficientNetB0(weights='imagenet')


def load_object_detection_model(model_path: str) -> Optional[Any]:
    """
    Laddar en objektdetektionsmodell från angiven sökväg.
    """
    try:
        model = load_model(model_path)
        logging.info(f"Modellen laddades framgångsrikt från {model_path}.")
        return model
    except Exception as e:
        logging.error(f"Misslyckades med att ladda modellen från {model_path}: {e}")
        return None


def detect_objects(model: Any, images: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detekterar objekt i bilder med den angivna modellen.
    """
    if model is None:
        logging.error("Ingen modell tillgänglig för objektdetektion.")
        return []
    try:
        predictions = model.predict(np.expand_dims(images, axis=0))
        detected_objects = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in predictions]
        return detected_objects
    except Exception as e:
        logging.error(f"Fel under objektdetektion: {e}")
        return []



def load_model_with_checks(model_path: str) -> Optional[Any]:
    """
    Laddar en modell med kontroller av filväg och filtyp.
    """
    if not isinstance(model_path, str) or not model_path.endswith('.h5'):
        raise ValueError(f"Ogiltig modellfil: {model_path}. Förväntar .h5 filändelse.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modellfilen {model_path} hittades inte.")
    try:
        model = load_model(model_path)
        logging.info(f"Modellen {model_path} laddades framgångsrikt.")
        return model
    except Exception as e:
        logging.error(f"Kunde inte ladda modellen från {model_path}: {e}")
        raise
    

def safe_read_image(path: str) -> Optional[np.ndarray]:
    """
    Läser säkert in en bild och kontrollerar bildtypen.
    """
    if not imghdr.what(path):
        logging.error(f"Filen {path} är inte en erkänd bildtyp.")
        return None
    try:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Bilden {path} kunde inte läsas.")
        return image
    except Exception as e:
        logging.error(f"Kunde inte läsa bild {path} på grund av: {e}")
        return None



def enhance_image_with_gan(model: Any, image: np.ndarray) -> np.ndarray:
    """
    Förbättrar en bild med en GAN-modell, anpassad efter bildanalys.
    """
    analysis_results = analyze_image(image)
    # Dynamisk anpassning baserat på analysresultat
    if analysis_results['detail_score'] > 0.5:
        # Använd en modellinställning som bevarar detaljer bättre
        model_setting = 'detail_preservation'
    else:
        model_setting = 'default'
    logging.info(f"Using {model_setting} setting for GAN enhancement.")
    enhanced_image = model.predict(np.expand_dims(image, axis=0))
    return np.squeeze(enhanced_image, axis=0)




def upscale_image_with_ai(model: Any, image_segment: np.ndarray) -> np.ndarray:
    """
    Dynamiskt uppskalar en bildsegment med AI baserat på segmentets storlek.
    """
    if image_segment.shape[0] < 256:
        try:
            # Dynamisk anpassning av uppskalningsfaktor
            logging.info("Performing AI upscaling.")
            upscaled_segment = np.squeeze(model.predict(np.expand_dims(image_segment, axis=0)), axis=0)
            return upscaled_segment
        except Exception as e:
            logging.error(f"Error during upscaling: {e}")
            raise RuntimeError(f"Upscaling failed due to an error: {e}")
    else:
        logging.info("Upscaling not required.")
        return image_segment



def analyze_image(image: np.ndarray) -> dict:
    """
    Analyserar en bild för att extrahera olika egenskaper.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray_image)
    brightness = np.mean(gray_image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv_image[:, :, 1])
    noise_level = estimate_noise_level(gray_image)
    detail_score = calculate_detail_score(image)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = np.mean(edges)
    return {
        'contrast': contrast,
        'brightness': brightness,
        'saturation': saturation,
        'noise_level': noise_level,
        'detail_score': detail_score,
        'edge_density': edge_density
    }


def estimate_noise_level(image: np.ndarray) -> float:
    """
    Implementera en metod för att uppskatta brusnivån i bilden.
    Detta kan baseras på variansen i en slätad version av bilden jämfört med originalet.
    """
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    diff_image = cv2.absdiff(image, blurred_image)
    noise_level = np.mean(diff_image)
    return noise_level


def calculate_detail_score(image: np.ndarray) -> float:
    """
    Beräknar en poäng som representerar mängden detaljer eller text i bilden.
    Detta kan använda edge detection eller textdetektionsalgoritmer.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    detail_score = np.mean(edges)
    return detail_score


def post_process(image: np.ndarray) -> np.ndarray:
    try:
        analysis_results = analyze_image(image)

        # Använd CLAHE för kontrastförbättring baserat på analysresultat
        if analysis_results['contrast'] < 50:
            image = apply_clahe(image, analysis_results['contrast'])

        # Dynamisk skärpning med hänsyn till detaljnivå och brus
        image = dynamic_sharpening(image, analysis_results)

        # Intelligent brusreducering baserat på uppskattad brusnivå
        if analysis_results['noise_level'] > 0.5:
            image = intelligent_noise_reduction(image, analysis_results['noise_level'])

        return image
    except Exception as e:
        logging.error(f"Advanced post-processing failed: {e}")
        return image


def apply_clahe(image: np.ndarray, contrast: float) -> np.ndarray:
    """
    Tillämpar CLAHE baserat på bildens kontrastnivå.
    """
    # Anpassa CLAHE-parametrar baserat på kontrastvärdet
    clip_limit = 2.0  # Exempelvärde, kan anpassas
    tile_grid_size = (8, 8)  # Exempelvärde, kan anpassas
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_image[:,:,0] = clahe.apply(lab_image[:,:,0])
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def dynamic_sharpening(image: np.ndarray, analysis_results: dict) -> np.ndarray:
    """
    Tillämpar dynamisk skärpning baserat på detaljnivå och eventuell närvaro av brus.
    """
    alpha = 1.5  # Skärpningsstyrka
    beta = 1.0 - alpha
    gamma = 0  # Justera enligt brusnivå och detaljpoäng
    
    # Justera dessa värden baserat på `analysis_results`
    if analysis_results['noise_level'] > 0.5:
        alpha -= 0.5  # Minska skärpningsstyrkan om bilden är brusig
    
    sharpened_image = cv2.addWeighted(image, alpha, cv2.GaussianBlur(image, (0, 0), 3), beta, gamma)
    return sharpened_image


def intelligent_noise_reduction(image: np.ndarray, noise_level: float) -> np.ndarray:
    # Anpassa styrkan i brusreduceringen baserat på `noise_level`
    strength = 10 if noise_level > 0.5 else 5
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)




def process_images_in_batch(image_paths: List[str], model: Any):
    """
    Bearbetar bilder i batcher för effektivitet, inklusive efterbearbetning och sparande.
    """
    datagen = ImageDataGenerator(rescale=1./255)

    # Antag att 'image_paths' är en mapp som innehåller bilder som ska bearbetas
    for image_path in image_paths:
        images = datagen.flow_from_directory(
            image_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode=None)

        predictions = model.predict(images)
        # Hantera förutsägelser för varje bild i batchen här


def process_image(input_image_path: str, output_image_path: str, model_path: str):
    """
    Bearbetar en bild genom att utföra objektdetektion, uppskalning och efterbearbetning.
    """
    try:
        model = load_object_detection_model(model_path)
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError("Misslyckades att läsa bild från sökvägen.")
    except Exception as e:
        logging.error(f"Fel vid bildbearbetning: {e}")
        return

    detected_objects = detect_objects(model, image)
    for (x1, y1, x2, y2) in detected_objects:
        object_segment = image[y1:y2, x1:x2]
        upscaled_segment = upscale_image_with_ai(model, object_segment)
        image[y1:y2, x1:x2] = upscaled_segment[:(y2-y1), :(x2-x1)]

    enhanced_image = post_process(image)
    cv2.imwrite(output_image_path, enhanced_image)
    logging.info("Bildförbättring genomförd och sparad.")

if __name__ == "__main__":
    input_image_path = 'din_bildväg.jpg'
    output_image_path = 'utgång_bildväg.jpg'
    model_path = 'din_modell_filväg.h5'
    process_image(input_image_path, output_image_path, model_path)

