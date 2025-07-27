import streamlit as st
from PIL import Image
import os
import numpy as np
from tensorflow.keras.models import load_model

# Función para cargar el modelo (con caché para mejor rendimiento)
@st.cache_resource
def cargar_modelo():
    ruta_modelo = "model/04_modelo.keras"
    id_drive = "1G3_ysyKP4uokQSnoIbcACWliRruUDSUx" 

    #Verificar si el modelo ya existe  
    if not os.path.exists(ruta_modelo):
        st.info("Descargando obteniendo el modelo ...")
        url = f"https://drive.google.com/uc?id={id_drive}"
        gdown.download(url, ruta_modelo, quiet=False)
        
    try:
        modelo = load_model(ruta_modelo)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None
    
    # Función para preprocesar la imagen
def preprocesar_imagen(imagen):
    """Preprocesa la imagen para que sea compatible con el modelo"""
    try:
        # Abrir imagen con PIL (Que es compatible con Streamlit y TensorFlow)
        img = Image.open(imagen)
        
        # Convertir a RGB si es necesario (por si la imagen tiene canal alpha)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar a 150x150 (igual que en el entrenamiento)
        img_resized = img.resize((150, 150))
        
        # Convertir a array de numpy
        img_array = np.array(img_resized)
        
        # Normalizar (dividir por 255.0 como en el entrenamiento)
        img_array = img_array / 255.0
        
        # Agregar dimensión de batch (el modelo espera un batch de imágenes)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None

# Función para hacer predicción
def predecir(modelo, imagen_procesada):
    """Realiza la predicción usando el modelo cargado"""
    try:
        # Hacer la predicción
        prediccion = modelo.predict(imagen_procesada)
        
        # Las clases están en el mismo orden que en tu entrenamiento
        labels = ['AD','CONTROL', 'PD'] #Azheimer, Control, Parkinson
        
        # Obtener la clase con mayor probabilidad
        clase_predicha_idx = np.argmax(prediccion[0])
        clase_predicha = labels[clase_predicha_idx]
        
        # Obtener la confianza (probabilidad máxima)
        confianza = np.max(prediccion[0])
        
        # Obtener todas las probabilidades para mostrar
        probabilidades = {labels[i]: prediccion[0][i] for i in range(len(labels))}
        
        return clase_predicha, confianza, probabilidades
    
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        return None, None, None

# Interfaz principal
st.markdown("<h1 style='text-align: center;'> Clasificador de Neuroimágenes</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Clasificación automática: Alzheimer (AD) | Control | Parkinson (PD)</p>", unsafe_allow_html=True)
st.markdown("---")

# Cargar el modelo al inicio
modelo = cargar_modelo()

if modelo is None:
    st.stop()  # Detener la ejecución si no se puede cargar el modelo

# Subida de imagen
imagen = st.file_uploader(
    " Subir Neuroimagen",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Formatos soportados: JPG, JPEG, PNG"
)

if imagen:
    # Botón para procesar
    if st.button("🔍 Analizar Imagen", type="primary"):
        with st.spinner(" Analizando neuroimagen..."):
            # Preprocesar la imagen
            imagen_procesada = preprocesar_imagen(imagen)
            
            if imagen_procesada is not None:
                # Hacer la predicción
                clase_predicha, confianza, probabilidades = predecir(modelo, imagen_procesada)
                
                if clase_predicha is not None:
                    # Mostrar resultados
                    st.markdown("---")
                    st.markdown("## **-> Resultados del Análisis**")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(imagen, caption="Neuroimagen Analizada", use_container_width=True)
                    
                    with col2:
                        # Determinar color según el resultado
                        if clase_predicha == "CONTROL":
                            color = "#4CAF50"  # Verde para control sano
                            #icon = "✅"
                        elif clase_predicha == "AD":
                            color = "#FF5722"  # Rojo para Alzheimer
                            #icon = "⚠️"
                        else:  # PD
                            color = "#FF9800"  # Naranja para Parkinson
                            #icon = "🔍" 
                        
                        # Resultado principal
                        st.markdown(f"""
                            <div style='
                                border: 2px solid {color};
                                border-radius: 15px;
                                padding: 20px;
                                background-color: #f9f9f9;
                                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
                                margin-bottom: 20px;
                            '>
                               <h3 style='color: #333; margin-bottom: 15px;'> Diagnóstico</h3>
                                <p style='font-size: 24px; margin: 10px 0;'><strong>Clase:</strong> <span style='color: {color}; font-weight: bold;'>{clase_predicha}</span></p>
                                <p style='font-size: 20px; margin: 10px 0;'><strong>Confianza:</strong> <span style='color: #2196F3; font-weight: bold;'>{confianza:.2%}</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Mostrar todas las probabilidades
                        st.markdown("### 📈 Probabilidades por clase:")
                        for clase, prob in probabilidades.items():
                            st.progress(float(prob), text=f"{clase}: {prob:.2%}")
                    
                    # Interpretación de resultados
                    st.markdown("---")
                    st.markdown("### Interpretación:")
                    
                    if clase_predicha == "CONTROL":
                        st.success(" **Control Sano**: La imagen sugiere patrones neurológicos normales.")
                    elif clase_predicha == "AD":
                        st.error(" **Alzheimer**: La imagen muestra patrones compatibles con enfermedad de Alzheimer.")
                    else:  # PD
                        st.warning(" **Parkinson**: La imagen presenta características asociadas con enfermedad de Parkinson.")
                    
                    st.info("**Nota importante**: Este es un análisis automatizado con fines informativos. Siempre consulte con un profesional médico para un diagnóstico definitivo.")

# Información adicional en la barra lateral
with st.sidebar:
    st.markdown("## Información del Modelo")
    st.markdown("""
    **Características:**
    - **Arquitectura:** ResNet50
    - **Resolución:** 150x150 píxeles
    - **Clases:**
                - *AD (Alzheimer)*
                - *Control (Sano)*
                - *PD (Parkinson)*
    - **Tipo:** Clasificación de neuroimágenes
    
    **Instrucciones:**
    1. Sube una neuroimagen
    2. Haz clic en "Analizar Imagen"
    3. Revisa los resultados
    
    **Formatos soportados:**
    - JPG, JPEG, PNG
    """)
    
    if modelo:
        st.success("Modelo cargado correctamente")
    else:
        st.error("Error al cargar el modelo")