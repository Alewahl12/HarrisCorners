import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lista de imagens
imagens = ['quarto_visao.jpeg', 'Sala02.jpg', 'Sala3.jpg']

# Lista de maxCorners que queremos testar
max_corners_list = [50, 100, 150]

for max_corners in max_corners_list:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 linha, 3 colunas

    for idx, (img_path, ax) in enumerate(zip(imagens, axes)):
        # Carregar a imagem em escala de cinza
        imagem_cinza = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if imagem_cinza is None:
            print(f"Erro ao carregar {img_path}")
            continue

        # Aplicar o detector Shi-Tomasi
        cantos = cv2.goodFeaturesToTrack(
            imagem_cinza,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=10
        )

        # Copiar a imagem para desenhar
        imagem_resultado = imagem_cinza.copy()

        if cantos is not None:
            cantos = np.round(cantos).astype(int)
            for canto in cantos:
                x, y = canto.ravel()
                cv2.circle(imagem_resultado, (x, y), 3, 255, -1)  # Círculo branco para marcar o canto

        # Mostrar no subplot
        ax.imshow(imagem_resultado, cmap='gray')
        ax.set_title(f'Imagem {idx+1}')
        ax.axis('off')

    fig.suptitle(f'Shi-Tomasi - maxCorners={max_corners}', fontsize=16)
    plt.tight_layout()
    plt.show()