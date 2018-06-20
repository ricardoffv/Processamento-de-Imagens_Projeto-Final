#Nome: Lucas Yudi Sugi 						Numero USP: 9293251
#Nome: Ricardo França Fernandes do Vale 	Numero USP: 9293477
#SCC 0251 - Processamento de Imagens - 2018/1o sem - Prof. Moacir Ponti
#Aplicando as tecnicas de segmentacao nas imagens do dataset
# -*- coding: utf-8 -*-

import numpy as np

#Size of face
sizeFace = 101

#Size of subImage
sizeSubImage = 25

threshold = 120

#Extract a sub image
def extractImage(img,size):
    
    #dimensions of image
    M = img.shape[0]
    N = img.shape[1]
    
    a = int((size-1)/2)
    b = int((size-1)/2)

    #Extract the sub image
    return np.copy(img[int((M/2))-a:int((M/2))+a+1,int((N/2))-b:int((N/2))+b+1])

#Populating the mask
def populateMask(mask,size,face):

    #dimensions of image
    M = mask.shape[0]
    N = mask.shape[1]
    
    a = int((size-1)/2)
    b = int((size-1)/2)
   
    #Setting mask
    mask[int((M/2))-a:int((M/2))+a+1,int((N/2))-b:int((N/2))+b+1] = face

    return mask

#Extract the mask that select the face
def extractMask(img,face,mean):
    
    #mask
    mask = np.zeros([img.shape[0],img.shape[1],3], dtype=np.uint8)

    #Growing the region that is close to the mean
    M = face.shape[0]
    N = face.shape[1]
    
    #Creating the mask
    for i in range(M):
        for j in range(N):
            colorR = np.absolute(face[i][j][0]-mean[0])
            colorG = np.absolute(face[i][j][1]-mean[1])
            colorB = np.absolute(face[i][j][2]-mean[2])
            color = np.sum(colorR+colorG+colorB)
            if(color < threshold):
                face[i][j] = 1
            else:
                face[i][j] = 0 

    return populateMask(mask,sizeFace,face)

#Apply the segmentation
def segmentation(img):
    
    #Face of person
    face = extractImage(img,sizeFace)
    
    #subImage that will be used for computed the mean colors
    subImage = extractImage(face,sizeSubImage)
    
    #mean RGB
    meanR = np.mean(subImage[:,:,0])
    meanG = np.mean(subImage[:,:,1])
    meanB = np.mean(subImage[:,:,2])
    
    #mask
    return extractMask(img,face,[meanR,meanG,meanB])


#a ideia e que a foto seja tirada com um fundo uniforme, de preferencia

#funcao que realiza o filtro abritario dados os pesos, a posicao de interesse e a matriz a ser trabalhada, 
#retornando o valor do pixel transformado
def abritary_filter (weights, x, y, image):
	n,m = weights.shape
	sub_matrix = np.zeros(weights.shape)
	#calculo dos indices de interesse na matriz da imagem
	a = int((n-1)/2)
	b = int((m-1)/2)
	#percursao na matriz de imagem, para gerar a submatriz, da operacao; para indices inexistentes mantem-se valor 0 na submatriz
	row = 0
	for i in range(x-a,x+a+1):
		col = 0
		for j in range(y-b,y+b+1):
			if(i < image.shape[0] and i >= 0 and j < image.shape[1] and j >= 0):
				sub_matrix[row,col] = image[i,j]
			col += 1
		row += 1

	#aplicacao do filtro
	weights_flip = np.flip(np.flip(weights, 0) ,1)

	#calculo e retorno do valor
	res = np.sum(np.multiply(sub_matrix, weights_flip))
	return res

#funcao que realiza a convolucao no dominio da frequencia os filtros, dada a imagem
def freq_domain_convolution(image, weights):
	#dominio de frequencia com filtragem arbitraria			
	res = np.zeros(image.shape, dtype=np.complex64)
	#calculo de F baseada na imagem
	F = np.fft.fft2(image)
	#calculo dos pesos W com vetor de tamanho igual o da imagem em que os valores antes inexistentes tornam-se zero
	reshaped_weights = np.zeros(image.shape, dtype=float)
	for i in range(weights.shape[0]):
		for j in range(weights.shape[1]):
			reshaped_weights[i,j] = weights[i,j]
	W = np.fft.fft2(reshaped_weights)
	#calculo da multiplicacao entre F e W
	res = np.multiply(W, F)
		
	#retorno do resultado
	return res

#calculo de celula do filtro
def laplacian_operation(x, y, std_deviation):
	return ((-1)/(np.pi*(std_deviation**4))) * (1 - ((x**2)+(y**2))/(2*(std_deviation**2))) * np.exp(-((x**2)+(y**2))/(2*(std_deviation**2)))

#funcao que realiza a convolucao no dominio da frequencia com um filtro gerado, dada a imagem, um tamanho de filtro e o desvio padrao
def laplacianfilter_convolution(image, n, std_deviation):
	#gerando coeficiente de andamento na matriz
	m = 10/(n-1)
	#criando range de linhas e colunas para o calculo
	indexes = np.arange(-5,5.00000001,m)
	indexes[n-1] = 5.0
	x_range = np.copy(indexes)
	y_range = np.copy(np.flip(indexes,0))
	#gerando o filtro LoG, aproveitando para pegar valores da normalizacao
	log_filter = np.zeros([n,n], dtype=float) 
	positive_numbers = 0.0
	negative_numbers = 0.0
	for i in range(n):
		for j in range(n):
			log_filter[i,j] = laplacian_operation(x_range[j], y_range[i], std_deviation)
			if(log_filter[i,j] >= 0):
				positive_numbers += log_filter[i,j]
			else:
				negative_numbers += log_filter[i,j]
	#normalizando o filtro
	for i in range(n):
		for j in range(n):
			if(log_filter[i,j] < 0):
				log_filter[i,j] = (-(positive_numbers/negative_numbers))*log_filter[i,j]

	#aplicando a convolucao
	res = np.zeros(image.shape, dtype=np.complex64)
	res = freq_domain_convolution(image, log_filter)
	return res

#funcao que realiza a convolucao no dominio da frequencia os filtros, dada a imagem
def space_domain_convolution(image, weights):
	res = np.zeros(image.shape, dtype=float)
	#simples aplicação dos pesos nos pixels
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			res[i,j] = abritary_filter(weights, i, j, image)
	return res

#funcao que retorna a imagem apos a operacao de sobel
def sobel_operation(image):
	#criando os filtros ja flipados
	fx = np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
	fy = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])

	#realizacao da convolucao para gerar novas imagens Ix e Iy
	Ix = space_domain_convolution(image, fx) 
	Iy = space_domain_convolution(image, fy) 

	#geracao de iout e, por fim, realizacao de sua transformada 2d
	res = np.zeros(image.shape, dtype=np.complex64)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			res[i,j] = ((Ix[i,j]**2)+(Iy[i,j]**2))**.5
	res = np.fft.fft2(res)
	
	return np.real(res)

#funcao para plotar imagem com configuracoes de visao
def plot(img):
	plt.subplot(111)
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	plt.show()


# #main
# #entrada do nome da imagem e geracao da mesma pelo imageio; esta e uma imagem do dataset
# #player used as example
# img = imageio.imread('./gea.jpg')

# #plotting orginal image
# # plot(img)

# #a extracao de fronteiras sera obtida na imagem em grayscale
# imageio.imwrite('./degea-gray.jpg', img[:, :, 0])
# gray_img = imageio.imread('./degea-gray.jpg')
# plot(gray_img)
# print(gray_img)

# #extracao de fronteiras
# #aplciando LoG
# #para tamanhos de filtros n=[3, 13]
# #sigma no intervalo [0,5; 2]
# #Justificativa: Assim, como regra, o tamanho de um filtro discreto LoG n × n deve ser 
# #projetado de modo que n seja o menor inteiro ímpar maior ou igual a 6*sigma
# # sigma = 0.2
# # for n in np.arange(3, 24, 2):
# # 	sigma = sigma + 0.3
# # 	iout = laplacianfilter_convolution(gray_img, n, sigma)
# # 	iout = (iout-iout.min()/iout.max()-iout.min())*255 
# # 	iout = iout.astype(np.uint8)
# # 	imageio.imwrite('./test.jpg', iout)
# # 	img_show = imageio.imread('./test.jpg')
# # 	plot(img_show)

# #aplicando sobel
# # iout = sobel_operation(gray_img)
# # #normalizacao e mudanca de tipagem para exibicao
# # iout = (iout-iout.min()/iout.max()-iout.min())*255 
# # iout = iout.astype(np.uint8)
# # imageio.imwrite('./test.jpg', iout)
# # img_show = imageio.imread('./test.jpg')
# # plot(img_show)

