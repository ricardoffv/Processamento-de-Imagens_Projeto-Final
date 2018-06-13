# SCC 0251 - Processamento de Imagens - Projeto-Final: e-Scout
Trabalho desenvolvido por Lucas Yudi Sugi (Numero USP: 9293251) e Ricardo França Fernandes do Vale (Numero USP: 9293477)

# Determinação de função em um time de futebol através de semelhanças faciais
Com a propagação dos testes de facebook que não utilizam nenhuma técnica sofisticada para determinar, por exemplo, "qual personagem de série você seria?", a ideia do projeto é, através de técnicas de aprendizado de características e análise de textura, com o auxílio de uma base previamente coletada, determinar qual função a pessoa teria em um time de futebol, dada uma foto da mesma como ponto de partida.
O projeto consiste em extrair características de imagens pré-estabelecidas, um dataset de 20 jogadores em cada uma das 6 posições diferentes (goleiro, zagueiro, lateral, meia, atacante e ponta), para comparar com as caractersticas de uma foto dada como entrada pelo usuário e, através de um algoritmo de classificação, indicar qual posição a pessoa da foto ocuparia em uma equipe de futebol.

## Imagens do Dataset
As imagens do dataset foram obtidas no site [FutWiz](https://www.futwiz.com/en/fifa18/worldcup/players). Tentou-se obter a maior diversidade possível de imagens em cada partição do dataset (o dataset é particionado pelas posições de um time de futebol). As imagens seguem um padrão em que o rosto do jogador está posicionado de frente para a câmera, com o fundo transparente, como [nesta foto](https://www.futwiz.com/assets/img/fifa18wc/faces/210257.png). 

## Metodologia
Tomando como base a discussão realizada com o professor, o livro Processamento Digital De Imagens - 3ª Ed. - 2011 gonzalez,Rafael C.; Woods,Richard E, e os slides da disciplina a solução do problema será definida nas seguintes etapas:

### Pré-processamento 
Como provavelmente as fotos tiradas pelo usuário não estarão em boas condições de luz, já que, o mesmo não está em um ambiente controlado, será necessário realizar uma etapa de pré-processamento que visa melhorar a qualidade da imagem para comparação. Além disso, esta etapa é necessária para que a imagem tenha melhores condições de ser processada na etapa de segmentação. As seguintes técnicas foram implementadas:

* Ajuste de intensidade:
  - Ajuste gamma
  - Equalização de histograma
* Suavização:
  - Filtro da média
* Aguçamento:
  - Laplaciano da gaussiana
  - Operador Sobel
  - High Boost
  - Filtering (Aceita qualquer filtro para convolução)
  
Tais técnicas serão combinadas e testadas, sendo que o melhor conjunto será utilizado.

### Redimensionamento 
Como as imagens do usuário terão uma resolução diferente comparada à da base de dados, é necessário realizar um redimensionamento das mesmas. Os seguintes passos são realizados para tal:

1. A imagem é recortada em altura e largura para que suas dimensões sejam proporcionais a 160 pixels (altura e largura das imagens da base de dados)
2. Com um filtro, realiza-se uma média dos pixels de modo que há um mapeamento de vários pixels da imagem original para a redimensionada

### Detecção/segmentação da face
Após ter a imagem em uma qualidade boa, será realizada uma detecção/segmentação da face com a finalidade de retirar o fundo da imagem. Para isso as seguintes técnicas serão testadas:
* Segmentação de imagens baseada em cor:
	- Extração de fronteiras
	- Regiões de Crescimento
	- Técnicas de Labelling

### Extração de características 
Com todas as outras etapas anteriores concluídas podemos extrair as características necessárias para posterior comparação. Testaremos as seguintes técnicas:
* Keypoint detector - Harris
* Orientation - HOG

### Comparação 
Para verificar qual a posição mais próxima do usuário será realizada o uso de algoritmos de aprendizado de máquina. Serão testados:
* Knn
* Dwnn
