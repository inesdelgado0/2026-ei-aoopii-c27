# 2026-ei-aoopii-c27

## Classificação de Atributos de Vestuário

### Grupo

**C27**

### Elementos

| Nome | Número | Email |
| --- | --- | --- |
| Filipa Calheiros | 31427 | filipacalheiros@ipvc.pt |
| Inês Delgado | 31414 | inesdelgado@ipvc.pt |

## Descrição do Projeto

Este projeto implementa uma pipeline de Visão por Computador baseada em classificação multi-rótulo, com o objetivo de automatizar a identificação e atribuição de etiquetas a produtos de vestuário no contexto do comércio eletrónico.

Ao contrário dos classificadores tradicionais de rótulo único, este modelo analisa uma única imagem de uma peça de roupa e prevê, em simultâneo, vários atributos pertencentes a diferentes categorias, como material, padrão, cor, estilo e outros detalhes visuais relevantes.

## Problema

No comércio eletrónico, a etiquetagem manual de produtos é um processo demorado, sujeito a erros humanos e, muitas vezes, inconsistente. Este sistema pretende fornecer uma saída estruturada e automática, que possa ser integrada em bases de dados de produtos, melhorando a pesquisa, a organização do catálogo e a experiência do utilizador.

| Campo | Descrição |
| --- | --- |
| Domínio | Fashion Tech / Visão por Computador |
| Tarefa | Classificação multi-rótulo, também designada por previsão de atributos |
| Principal desafio | Distinguir características visuais subtis, como riscas finas vs. padrão liso, ou estilo formal vs. smart casual, em imagens com diferentes condições de iluminação, poses e enquadramentos |

## Tecnologias Utilizadas

O projeto é desenvolvido com uma stack moderna de aprendizagem profunda, adequada a tarefas de classificação de imagens.

| Categoria | Tecnologia |
| --- | --- |
| Linguagem | Python 3.10+ |
| Framework de Deep Learning | PyTorch |
| Arquiteturas de Modelo | ResNet-50 e Vision Transformer (ViT) |
| Processamento de Dados | OpenCV, Pillow e NumPy |
| Dataset | DeepFashion, no subconjunto de previsão de atributos |

### Arquiteturas

- **ResNet-50:** usada como backbone para extração eficiente de características espaciais, texturas e padrões visuais.
- **Vision Transformer (ViT):** arquitetura opcional, usada para captar relações globais na imagem e contexto visual associado ao estilo.

## Dataset

O projeto utiliza o **DeepFashion**, mais especificamente o subconjunto de previsão de atributos, que contém mais de 289 000 imagens com anotações profissionais.
