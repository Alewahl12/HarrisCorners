import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lista de imagens
imagens = ['quarto_visao.jpeg', 'Sala02.jpg', 'Sala3.jpg']

# Lista de ksize que queremos testar
ksize_list = [3, 5, 7]

for ksize in ksize_list:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 linha, 3 colunas

    for idx, (img_path, ax) in enumerate(zip(imagens, axes)):
        # Carregar a imagem em escala de cinza
        imagem_cinza = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if imagem_cinza is None:
            print(f"Erro ao carregar {img_path}")
            continue

        # Aplicar o detector de Harris
        harris = cv2.cornerHarris(
            imagem_cinza, 
            blockSize=2, 
            ksize=ksize, 
            k=0.04
        )

        # Dilatar para realçar os cantos
        harris = cv2.dilate(harris, None)

        # Copiar imagem para desenhar
        imagem_resultado = imagem_cinza.copy()

        # Marcar os cantos detectados
        imagem_resultado[harris > 0.01 * harris.max()] = 255

        # Mostrar no subplot
        ax.imshow(imagem_resultado, cmap='gray')
        ax.set_title(f'Imagem {idx+1}')
        ax.axis('off')

    fig.suptitle(f'Harris Corner Detection - ksize={ksize}', fontsize=16)
    plt.tight_layout()
    plt.show()