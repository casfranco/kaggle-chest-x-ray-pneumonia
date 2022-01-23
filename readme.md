# Detecção de pneumonia
Cesar A. Sierra Franco <br>
[@casfranco](https://twitter.com/casfranco)


O projeto foi desenvolvido visando o treinamento de modelos de classificação de imagem. Como caso de uso foi utilizado um dataset de classificação de imagens raio-x de torax para detecção de pneumonia. O dataset se encontra disponível de forma pública no site de Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

A seguir as principais caracteristicas do sistema

- Tipo de problema: classificação binaria
- Entrada: Imagem de raio-X pulmão - array (1024x1024,1)
- Saída: predição de pneumonia (0=Normal, 1=Pneumonia)
- Dataset composto por 5840 imagens anotadas
- Modelos baseados em redes neurais
- Avaliação usando métricas de classificação.

Seguir o notebook `main_train.ipynb` com as explicações do passo a passo de geração de resultados.

## Dataset

Para a execução deste notebook python é necessario baixar o dataset do site de kaggle, localizando os arquivos na pasta `data/datasets/` e a instalação das bibliotecas especificadas no arquivo `readme.txt`

```bash
└── data
    └── datasets
        └── chest_xray
            ├── train
            │   ├── NORMAL
            │   ├── PNEUMONIA
            ├── val
            │   ├── NORMAL
            │   ├── PNEUMONIA
            ├── test
            │   ├── NORMAL
            │   ├── PNEUMONIA                    
```
## Processo de treinamento e validação

Para execução do processo de treinamento e avaliação passo a passo, execute o jupyter notebook: main_train. Adicionalmente foi fornecido um script para execução do pipeline a ser executado desde a raiz do projeto.

```bash
python main_train.py
```

### Arquivo de configuração
Para facilitar a configuração do processo de treinamento e ajuste de hiperparâmetros foi fornecido um arquivo de configuração JSON localizado em `data/config_files/cls_params.json`

```yaml
{
    "classes": ["normal", "pneumonia"],
    "n_classes":1,
    "model": "resnet18",
    "optimizer": "Adam",
    "loss":"binary_crossentropy",
    "batch_size": 128,
    "num_epochs": 40,
    "learning_rate": 0.001,
  
    "image_height": 128,
    "image_width": 128,
    "channels": 1,
  
    "apply_data_augmentation": false,
    "data_augmentation": {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.1,
        "horizontal_flip": true,
        "vertical_flip": false,
        "fill_mode" : "constant"
      },

    "evaluate_validation_metrics":true,
    "evaluate_test_metrics":false,

    "save_model_results": true,
    "save_prefix":""

}
```

### Criação de modelos e loop de treinamento


As arquiteturas de rede neural e o processo de treinamento será apoiado com o uso da biblioteca `tensorflow`. Os métodos de criação, compilação, treinamento, avaliação e persistencia de resultados foram encapsulados nas classes `ModelTrainer` e `ModelManager`. Será possível obervar durante a execução do processo de treinamento a evolução na função de perda e na acuracia para cada episódio de treinamento gerando as chamadas "curvas de aprendizado". A avaliação destas curvas permitem determinar se ocorre "overfitting" ou "underfitting" para o modelo em processo de treinamento.<br>

Para criação das arquiteturas de rede neural, a classe `ModelManager` integra funções das bibliotecas [efficientnet](https://github.com/qubvel/efficientnet) e [classification models](https://github.com/qubvel/classification_models). Com isto poderão ser diretamente criados varias arquiteturas do estado da arte modificando o campo model no arquivo de configuração. Dentre alguns dos modelos suportados se encontram:

- resnet18, resnet34, densenet169, mobilenet.
- EfficientNetB0, ..., EfficientNetB7

Para mais informações recomenda-se observar os modelos suportados pelas bibliotecas mencionadas.<br>

Adicionalmente, a classe `ModelManager` suporta a criação de modelos customizados via `tensorflow.keras`. Como exemplo, integrou-se um modelo sequencial customizado que mostrou bons resultados nas métricas de classificação e tempo de processamento. Para utilizar o modelo sequencial deve se utilizar o nome `custom_model` no campo model de arquivo de configuração.

### Geração de Reportes sobre o conjunto de validação

Na finalização do processo de treinamento, serão computadas as métricas de avaliação no dataset de validação. Com isto será possível fazer ajustes nos hyperparâmetros de treinamento até obter os resultados desejados. As metricas computadas são as seguintes:
- Acurácia
- F1
- Precisão
- Recal
- Curvas de aprendizado
- Matriz de confusão
- Curva ROC

A execução de treinamento gera e armazena automaticamente resultados para cada experimento executado refentes ao historico de treinamento, métricas de avaliação e melhor modelo na validação (formato h5) na pasta `data/results/`. Cada execução terá então como resultado a seguinte estrutura. 

```bash
└── data
    └── results
        └── chest_xray
            ├── custom_model_2022_0120_1633
            │   ├── history_results
            │   ├── metrics_evaluation
            │   ├── trained_model                    
```

### Comparação entre experimentos
Quando realizado o processo de treinamento de varios modelos de classificação, os resultados são armazenados em disco para futuras comparações. Com este fim a classe classe `TrainedReports` é a responsável de realizar ditas comparações entre as execuções. <br>

Para selecionar candidatos de comparação, deverá se fornecer uma lista com os nomes das pastas geradas como o resultado do treinamento. 

Para maiores detalhes observar a execução das funções no Jupyter Notebook incluido como demo.
