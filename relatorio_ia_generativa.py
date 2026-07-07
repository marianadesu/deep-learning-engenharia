# Importa a biblioteca os para ler variaveis de ambiente.
# Ela sera usada para verificar se existe uma chave de API configurada.
import os

# Importa a biblioteca da OpenAI.
# Ela so sera usada se voce tiver configurado a chave OPENAI_API_KEY.
from openai import OpenAI


# Dados do equipamento analisado.
# Aqui usamos o mesmo exemplo gerado pela rede neural.
equipment_data = {
    "corrente": 58,
    "tensao": 214,
    "temperatura": 96,
    "vibracao": 5.2,
    "dias_sem_manutencao": 145,
    "probabilidade_falha": 92.16,
    "classificacao": "risco de falha",
}


# Esta funcao monta o prompt que sera enviado para a IA generativa.
def create_prompt(data):
    prompt = f"""
Voce e um engenheiro de manutencao industrial.

Gere um relatorio tecnico curto com base nos dados abaixo:

Corrente: {data["corrente"]} A
Tensao: {data["tensao"]} V
Temperatura: {data["temperatura"]} C
Vibracao: {data["vibracao"]} mm/s
Dias desde a ultima manutencao: {data["dias_sem_manutencao"]}
Probabilidade estimada de falha: {data["probabilidade_falha"]}%
Classificacao: {data["classificacao"]}

O relatorio deve conter:
1. Diagnostico do equipamento.
2. Possiveis causas do risco.
3. Recomendacoes de manutencao.
4. Nivel de urgencia.

Use linguagem tecnica, clara e objetiva.
"""
    return prompt


# Esta funcao usa IA generativa para criar o relatorio tecnico.
def generate_report_with_ai(prompt):
    # Cria o cliente da OpenAI.
    # A chave da API deve estar salva como OPENAI_API_KEY no GitHub Secrets.
    client = OpenAI()

    # Envia o prompt para o modelo e recebe a resposta gerada.
    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    # Retorna apenas o texto gerado pela IA.
    return response.output_text


# Esta funcao cria um relatorio simples sem API.
# Ela serve como alternativa caso a chave da API nao esteja configurada.
def generate_report_without_api(data):
    report = f"""
RELATORIO TECNICO DE MANUTENCAO

Diagnostico:
O equipamento apresenta {data["classificacao"]}, com probabilidade estimada de falha de {data["probabilidade_falha"]}%.

Possiveis causas:
A temperatura de {data["temperatura"]} C, a vibracao de {data["vibracao"]} mm/s e o periodo de {data["dias_sem_manutencao"]} dias sem manutencao indicam uma condicao operacional critica.

Recomendacoes:
Recomenda-se realizar inspecao preventiva, verificar rolamentos, conexoes eletricas, sistema de ventilacao e historico de manutencao do equipamento.

Nivel de urgencia:
Alto. O equipamento deve ser avaliado antes de continuar operando por longos periodos.
"""
    return report


def main():
    # Cria o prompt com os dados do equipamento.
    prompt = create_prompt(equipment_data)

    print("Gerando relatorio tecnico com IA Generativa...\n")

    # Verifica se a chave da OpenAI existe.
    # Se existir, usa IA generativa real.
    # Se nao existir, usa um relatorio automatico simples.
    if os.getenv("OPENAI_API_KEY"):
        report = generate_report_with_ai(prompt)
    else:
        print("OPENAI_API_KEY nao encontrada.")
        print("Gerando relatorio local sem chamada de API.\n")
        report = generate_report_without_api(equipment_data)

    print(report)


# Executa o programa.
if __name__ == "__main__":
    main()
