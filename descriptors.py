#Nome: Lucas Yudi Sugi 						Numero USP: 9293251
#Nome: Ricardo França Fernandes do Vale 	Numero USP: 9293477
#SCC 0251 - Processamento de Imagens - 2018/1o sem - Prof. Moacir Ponti

import numpy as np 

#calculo da correlacao da matriz de coocorrencia, incluindo calculos auxiliares
#sao passados por parametro a matriz de coocorrencia e o vetor com as intensidades presentes na matriz
def coocurrence_correlation(G, unique):
	correlation = 0.0
	#calculos auxiliares
	mean_i = mean_j = std_i = std_j = 0.0
	#media na direcao das colunas
	counter = 0
	for i in unique:
		mean_i = mean_i + (i*np.sum( G[counter, :] ))
		counter += 1
	#media na direcao das linhas
	counter = 0
	for j in unique:
		mean_j = mean_j + (j*np.sum( G[:, counter] ))
		counter += 1
	#desvio padrao na direcao das linhas
	counter = 0
	for i in unique:
		std_i = std_i + (((i - mean_i)**2)*np.sum( G[counter, :] ))
		counter += 1
	#desvio padrao na direcao das colunas
	counter = 0
	for i in unique:
		std_j = std_j + (((j - mean_j)**2)*np.sum( G[:, counter] ))
		counter += 1

	#calculo da correlacao em si
	if (std_i > 0 and std_j > 0):
		for i in range(G.shape[0]):
			for j in range(G.shape[1]):
				correlation = correlation + ((int(unique[i]) - mean_i)*(int(unique[j]) - mean_j))*G[i,j] 
		correlation = correlation/(std_i*std_j)

	return correlation	

#gerando os cinco descritores de textura da imagem de parametro
def texture_descriptors(img):
	C = 256

	#pegando as intensidades na imagem e fazendo mapeamento reverso, ou seja, 
	#pegando o indice da posicao em que esta a determinada intensidade 
	unique = np.unique(img)
	intensity = np.zeros(C, dtype=np.uint8)
	for key in range(len(unique)):
		intensity[unique[key]] = key

	#criando matriz esparsa para a matriz de coocorrencia G
	#isso tornara o processo mais rapido e ira computar os calculos com a mesma precisao
	#pois as posicoes nao informadas serao zero e nao acrescentarao informacao adicional para G
	G = np.zeros([len(unique), len(unique)])

	#percorrendo a matriz para verificar os deslocamentos e atualizar G
	#cada vez que um elemento aij tiver como correspondente no deslocamento Q(1,1) bij
	#sera acrescido 1 na matriz G na posicao aij-bij
	for i in range(img.shape[0]-1):
		for j in range(img.shape[1]-1):
			G[ intensity[img[i,j]] ][ intensity[img[i+1,j+1]] ] += 1

	#normalizacao de G
	G = G/np.sum(G)

	#obtencao dos descritores de haralick baseados na matriz G
	#energia
	energy = np.sum(G**2)

	#entropia
	entropy = - np.sum(G*(np.log(G+0.001)/np.log(2)))
	
	#contraste
	contrast = 0.0
	for i in range(G.shape[0]):
		for j in range(G.shape[1]):
			contrast = contrast + float((int(unique[i])-int(unique[j]))**2)*G[i,j]
	contrast = contrast/((C-1)**2)

	#correlacao
	correlation = coocurrence_correlation(G, unique)

	#homogeneidade
	homogeneity = 0.0
	for i in range(G.shape[0]):
		for j in range(G.shape[1]):
			homogeneity = homogeneity + G[i,j]/(abs(int(unique[i])-int(unique[j]))+1)
	
	#retorno do vetor de textura
	return np.array([energy, entropy, contrast, correlation, homogeneity])

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
			if(i < image.shape[0] and i >= 0 and j < image.shape[0] and j >= 0):
				sub_matrix[row,col] = image[i,j]
			col += 1
		row += 1

	#aplicacao do filtro
	weights_flip = np.flip(np.flip(weights, 0) ,1)

	#calculo e retorno do valor
	res = np.sum(np.multiply(sub_matrix, weights_flip))
	return res

#funcao que realiza a convolucao no dominio da frequencia os filtros, dada a imagem
def space_domain_convolution(image, weights):
	res = np.zeros(image.shape, dtype=float)
	#simples aplicação dos pesos nos pixels
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			res[i,j] = abritary_filter(weights, i, j, image)
	return res

#funcao que retorna as imagens gx e gy apos a operacao de sobel
def sobel_operation(image):
	#criando os filtros ja flipados
	fx = np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
	fy = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])

	#realizacao da convolucao para gerar novas imagens Ix e Iy
	gx = space_domain_convolution(image, fx) 
	gy = space_domain_convolution(image, fy) 
	
	return [gx, gy]

#gerando a descricao de gradiente da imagem de parametro
def gradient_descriptors(img):
	#geracao das imagens gx e gy
	[gx, gy] = sobel_operation(img)

	#ignorando as bordas da imagem
	gx = gx[1:(gx.shape[0]-1), 1:(gx.shape[1]-1)]	
	gy = gy[1:(gy.shape[0]-1), 1:(gy.shape[1]-1)]	

	#calculo da magnitude
	M = np.sqrt((gx**2)+(gy**2))/np.sum(np.sqrt((gx**2)+(gy**2)))

	#calculo do histograma dos angulos 
	angles = (np.arctan2(gy, gx) * 180.0 / np.pi)+180.0
	#no histograma sao somadas as magnitudes de angulos em 18 bins que contem os angulos de interesse
	histogram = np.zeros(18, dtype=float)
	for i in range(angles.shape[0]):
		for j in range(angles.shape[1]):
			if (angles[i,j] == 360):
				histogram[0] = histogram[0] + M[i,j]
			else:
				histogram[int(angles[i,j]/20)] = histogram[int(angles[i,j]/20)] + M[i,j]

	#retorna-se o vetor de bins com as magnitudes
	return histogram