import os
import requests
import urllib.parse

def descargar_documentos(urls_file="urls.txt", download_dir="docs"):
    """
    Descarga documentos desde las URLs listadas en un archivo de texto.

    Args:
        urls_file (str): Ruta del archivo que contiene las URLs (una por línea).
        download_dir (str): Directorio donde se guardarán los documentos.
    """
    
    # Verificar y crear el directorio de descarga si no existe
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    # Leer las URLs desde el archivo
    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    for url in urls:
        filename = url.split("/")[-1]
        filename = urllib.parse.unquote(filename)
        file_path = os.path.join(download_dir, filename)
        if os.path.exists(file_path):
                print(f"'{filename}' ya existe. Saltando...")
                continue 
        try:
            response = requests.get(url, headers=headers)
            
            # Excepción si la respuesta no es 200 (OK)
            if response.status_code != 200:
                print(f"Error al descargar '{url}': {response.status_code}")
                continue
            
            # Obtener el nombre del archivo desde Content-Disposition o desde la URL
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition:
                filename = content_disposition.split("filename=")[-1].strip('\"')
            file_path = os.path.join(download_dir, filename)
            
            # Verificar si el archivo ya existe
            if os.path.exists(file_path):
                print(f"'{filename}' ya existe. Saltando...")
                continue
            
            # Guardar el archivo
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            print(f"Se ha descargado '{filename}'")
        
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión al intentar descargar '{url}': {e}")

