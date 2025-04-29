import cv2

imagem = cv2.imread("Sala02.jpg", cv2.IMREAD_GRAYSCALE)

# Criar o objeto SIFT
sift = cv2.SIFT_create()

# Detectar e computar características
keypoints, descriptors = sift.detectAndCompute(imagem, None)

# Desenhar keypoints na imagem
imagem_sift = cv2.drawKeypoints(imagem, keypoints, None)

# Exibir resultados
cv2.imshow('SIFT', imagem_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()
