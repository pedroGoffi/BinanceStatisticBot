# Sistema Inteligente de Análise de Criptomoedas

## Visão Geral
Este sistema utiliza a API da Binance para coletar dados históricos de criptomoedas, calcular indicadores técnicos como MACD, RSI, Média Móvel e Z-Score, e prever tendências de preços. Ele permite tomar decisões baseadas em dados em tempo real para comprar ou vender criptomoedas, ajudando os investidores a fazerem escolhas informadas.

## Funcionalidades Principais:
- **Coleta de Dados Históricos**: Utiliza a API da Binance para capturar dados históricos de criptomoedas.
- **Cálculo de Indicadores Técnicos**: Determina MACD, RSI, Média Móvel (MA50 e MA200), e Z-Score.
- **Avaliação de Suporte e Resistência**: Avalia os níveis de suporte e resistência para ajudar na análise técnica.
- **Previsão de Tendência com Z-Score**: Utiliza Z-Score para prever a tendência de preço da criptomoeda.
- **Decisão de Compra ou Venda**: Baseado numa combinação de indicadores técnicos, decide se deve comprar ou vender uma criptomoeda.

## Dependências:
- `binance`: para acesso à API da Binance.
- `pandas`: para manipulação e análise de dados.
- `numpy`: para cálculos matemáticos.
- `scipy.stats.zscore`: para calcular o Z-Score das taxas de fechamento.
- `multiprocessing`: para processamento paralelo para aumentar a eficiência.

## Instalação
1. Clone o repositório:
   ```bash
   git clone [URL_DO_REPOSITORIO]
   ```
2. Navegue até o diretório do projeto:
   ```bash
   cd [NOME_DO_PROJETO]
   ```
3. Crie um arquivo `.env` no diretório raiz do projeto com as seguintes variáveis de ambiente:
   ```plaintext
   API_KEY=SEU_API_KEY
   API_SECRET=SEU_API_SECRET
   ```
4. Instale as dependências usando pip:
   ```bash
   pip install -r requirements.txt
   ```

## Uso
Para executar o sistema e começar a análise das criptomoedas, basta executar o script principal:
```bash
python main.py
```

## Estrutura do Código
- **main.py**: Script principal que inicializa a conexão API, coleta dados e executa as funções de cálculo.
- **crypto_analysis.py**: Módulo que contém todas as funções de cálculo de indicadores técnicos e análise de suporte/resistência.


## Exemplo de Uso
O script executa uma análise automática das criptomoedas com base nos dados coletados e retorna uma lista de criptomoedas com potencial de ganho alto (superior ao limite de `GAIN_THRESHOLD`).

```python
# Exemplo de chamada para análise
resultados = check_coin_state()
for crypto in resultados:
    print_coin_details(crypto)
```

## Documentação dos Modos de Análise
- **`should_buy()`**: Determina se é hora de comprar uma criptomoeda baseado nos critérios técnicos.
- **`await_till_value()`**: Espera até que as condições técnicas sejam favoráveis para uma ação de compra.
- **`should_sell()`**: Determina se é hora de vender uma criptomoeda baseado nos critérios técnicos.

## Considerações
- Ajuste o `GAIN_THRESHOLD` conforme necessário para definir o limite de ganho mínimo aceitável.
- Certifique-se de substituir os valores `API_KEY` e `API_SECRET` com suas próprias credenciais da Binance.

## Licença
Este projeto está licenciado sob a MIT License. Consulte o arquivo LICENSE para mais informações.

## Autor
Pedro Henrique Goffi de Paulo


Este README fornece uma visão geral clara do sistema, detalhando desde a configuração até as funcionalidades principais e exemplos de uso, facilitando a compreensão e implementação do projeto.
