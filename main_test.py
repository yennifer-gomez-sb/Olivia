import pandas as pd
import time
#from src.preprocess_data.preprocess_test import process_csv
from src.model.calificador import calificador
#from src.model.clasificador_topicos import clasificador_topico 


def process_and_calify_csv(input_csv_path, output_excel_path):
    """
    Procesa un archivo CSV aplicando las funciones `calificador` y `clasificador_topico` a cada fila,
    y guarda los resultados en un archivo Excel.
    
    :param input_csv_path: Ruta del archivo CSV de entrada.
    :param output_excel_path: Ruta del archivo Excel de salida.
    """
    # Preprocesar el archivo CSV
    #process_csv(input_csv_path, input_csv_path)

    # Leer el archivo CSV procesado
    df = pd.read_excel(input_csv_path)

    # Verificar y crear columnas necesarias si no existen
    if 'calificacion_ailab' not in df.columns:
        df['calificacion_ailab'] = ''
    if 'context' not in df.columns:
        df['context'] = ''
    if 'topico' not in df.columns:
        df['topico'] = ''

    # Iterar sobre cada fila del DataFrame y aplicar las funciones
    for index, row in df.iterrows():
        question = row['input']  # Nombre correcto de la columna
        answer = row['output_ailab']   # Nombre correcto de la columna

        # Aplicar la función calificador
        try:
            result_tuple = calificador(question, answer)

            if len(result_tuple) == 2:
                parsed_result, context = result_tuple
                calificacion = parsed_result.get('calificacion', '')
                print(f"Fila {index} - Calificación: {calificacion}")
                
                df.at[index, 'calificacion_ailab'] = calificacion
                df.at[index, 'context'] = context
            else:
                raise ValueError("La función 'calificador' no devolvió el número esperado de valores.")
        except ValueError as e:
            print(f"Error en la fila {index} al calificar: {e}")
            df.at[index, 'calificacion_ailab'] = 'Error'
            df.at[index, 'context'] = ''
        except Exception as e:
            print(f"Ocurrió un error inesperado en la fila {index} al calificar: {e}")
            df.at[index, 'calificacion_ailab'] = 'Error'
            df.at[index, 'context'] = ''

        # Pausa entre iteraciones
        time.sleep(0.1)

    # Guardar el DataFrame modificado en un archivo Excel
    df.to_excel(output_excel_path, index=False)
    print(f"El archivo Excel se guardó correctamente en {output_excel_path}")

# Ejemplo de uso
input_csv = '/home/yennifer-gomez/ailab-olivia-faqs/data/evaluacion/AC_test_ailab_gemflash.xlsx'
output_excel = '/home/yennifer-gomez/ailab-olivia-faqs/data/evaluacion/AC_testeo_ailab_gemflash.xlsx'

process_and_calify_csv(input_csv, output_excel)