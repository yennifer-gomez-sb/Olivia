import pandas as pd
import re

def split_content(content):
    """
    Extrae la sección de PREGUNTA y RESPUESTA del contenido.
    """
    match = re.search(r'(PREGUNTA:.*?)(RESPUESTA:.*)', content, re.DOTALL)
    if match:
        pregunta = match.group(1).strip()
        respuesta = match.group(2).strip()
    else:
        pregunta = content.strip()
        respuesta = None
    return pregunta, respuesta

def clean_content(pregunta, respuesta):
    """
    Limpia la pregunta y respuesta removiendo secuencias especiales.
    """
    def remove_special_characters(text):
        # Eliminar backslashes, saltos de línea y comillas
        text = re.sub(r'\\n|\\|\"', '', text)
        return text.strip()
    
    if pregunta:
        pregunta = re.sub(r'^PREGUNTA:\s*', '', pregunta, flags=re.IGNORECASE).strip()
        pregunta = remove_special_characters(pregunta)
    
    if respuesta:
        respuesta = re.sub(r'^RESPUESTA:\s*', '', respuesta, flags=re.IGNORECASE).strip()
        respuesta = remove_special_characters(respuesta)
    
    return pregunta, respuesta

def contains_ac(metadata):
    return '"ac"' in metadata if pd.notnull(metadata) else False

def contains_sb(metadata):
    return '"sb"' in metadata if pd.notnull(metadata) else False

def process_and_save_ac_sb(input_excel_path, sheet_name='01082024', ac_output_path='./data/data_ac.csv', sb_output_path='./data/data_sb.csv', expand=0):
    # Cargar el archivo raw
    df_raw = pd.read_excel(input_excel_path, sheet_name=sheet_name)
    
    # Separar Pregunta y Respuesta a partir de 'Content'
    df_raw[['Pregunta', 'Respuesta']] = df_raw['Content'].apply(lambda x: pd.Series(split_content(x)))
    
    # Conservar solo columnas relevantes
    df_processed = df_raw[['Pregunta', 'Respuesta', 'Metadata']].copy()
    
    # Limpiar contenido de Pregunta y Respuesta
    df_processed['Pregunta'], df_processed['Respuesta'] = zip(*df_processed.apply(
        lambda row: clean_content(row['Pregunta'], row['Respuesta']), axis=1))
    
    # Si expand es 1, expandir las preguntas
    if expand == 1:
        expanded_rows = []
        for _, row in df_processed.iterrows():
            # Dividir las preguntas utilizando el carácter '¿'
            questions = row['Pregunta'].split('¿')
            questions = [q.strip() for q in questions if q.strip()]
            
            # Generar una fila por cada pregunta
            for q in questions:
                new_row = row.copy()
                # Asegurarnos de que cada pregunta empiece con '¿'
                if not q.startswith('¿'):
                    q = '¿' + q
                new_row['Pregunta'] = q
                expanded_rows.append(new_row)
        
        df_processed = pd.DataFrame(expanded_rows)
        df_processed = df_processed.reset_index(drop=True)
    
    # Agregar columna 'ids'
    df_processed.insert(0, 'ids', df_processed.index.astype(str))
    
    # Filtrar df_ac y df_sb
    df_ac = df_processed[df_processed['Metadata'].apply(contains_ac)].copy()
    df_sb = df_processed[df_processed['Metadata'].apply(contains_sb)].copy()
    
    # Reiniciar los ids en df_ac y df_sb
    df_ac['ids'] = range(len(df_ac))
    df_sb['ids'] = range(len(df_sb))
    
    # Guardar los resultados finales en CSV
    df_ac.to_csv(ac_output_path, index=False)
    df_sb.to_csv(sb_output_path, index=False)

    print("Archivos guardados:")
    print(f"AC: {ac_output_path}")
    print(f"SB: {sb_output_path}")


# Sin expansión:
# process_and_save_ac_sb('./data/raw.xlsx', '01082024', './data/expanded/data_ac.csv', './data/expanded/data_sb.csv', expand=0)

# Con expansión:
process_and_save_ac_sb('./data/raw.xlsx', '01082024', './data/expanded/data_ac_I.csv', './data/expanded/data_sb.csv', expand=1)
