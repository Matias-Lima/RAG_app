import time
import streamlit as st
from document_processing import cria_chain_conversa, PASTA_ARQUIVOS
from io import BytesIO
from PIL import Image
import pandas as pd

def reset_session_state():
    # Limpar as variáveis do estado da sessão
    st.session_state['a'] = []
    st.session_state['b'] = []
    st.session_state['c'] = []
    st.session_state['d'] = []
    st.session_state.pop('chain', None)  # Remove o chain se existir

def sidebar():
    uploaded_files = st.file_uploader(
        'Adicione seus arquivos', 
        type=['pdf', 'docx', 'epub', 'xlsx', 'xls', 'html'], 
        accept_multiple_files=True
    )
    
    if uploaded_files is not None:
        # Remover arquivos antigos na pasta
        for arquivo in PASTA_ARQUIVOS.glob('*'):
            arquivo.unlink()

        # Salvar arquivos enviados
        for file in uploaded_files:
            with open(PASTA_ARQUIVOS / file.name, 'wb') as f:
                f.write(file.read())
    
    # Botão de inicialização ou atualização do chatbot
    label_botao = 'Inicializar ChatBot'
    if 'chain' in st.session_state:
        label_botao = 'Atualizar ChatBot'

    if st.button(label_botao, use_container_width=True):
        # Limpar variáveis e estado da sessão
        reset_session_state()

        # Verificar se há arquivos na pasta
        if len(list(PASTA_ARQUIVOS.glob('*'))) == 0:
            st.error('Adicione arquivos para inicializar o chatbot')
        else:
            st.success('Inicializando o ChatBot...') 
            st.session_state['a'], st.session_state['b'], st.session_state['c'], st.session_state['d'] = cria_chain_conversa()
        st.rerun()

def inform():
    a = st.session_state.get('a', [])
    b = st.session_state.get('b', [])
    c = st.session_state.get('c', [])
    d = st.session_state.get('d', [])

    st.write("Vector Store Salvo ------")
    st.write(d)

    st.write("Texto extraídos:")
    st.write(a)

    if b:
        st.write("Imagens extraídas:")
        for i, image in enumerate(b):
            st.image(image, caption=f"Imagem {i + 1}")
    
    # Exibindo as tabelas (caso queira mostrar as tabelas também)
    if c:
        st.write("Tabelas extraídas:")
        for i, table in enumerate(c):
            st.write(f"Tabela {i + 1}:")
            
            if table:
                # Mostra a tabela em formato de DataFrame interativo
                st.dataframe(table)  # Usando dataframe para exibição interativa
            else:
                # Caso a tabela não seja um DataFrame, exibe a tabela como JSON (ou faça outro tratamento adequado)
                st.write(table)

def chat_window():
    st.header('🤖 Bem-vindo ao Chat com PDFs, Docx, html and epub', divider=True)

    if not 'chain' in st.session_state:
        st.error('Faça o upload de Arquivos para começar!')
        st.stop()
    
    chain = st.session_state['chain']
    memory = chain.memory

    mensagens = memory.load_memory_variables({})['chat_history']

    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    nova_mensagem = st.chat_input('Converse com seus documentos...')
    if nova_mensagem:
        chat = container.chat_message('human')
        chat.markdown(nova_mensagem)
        chat = container.chat_message('ai')
        chat.markdown('Gerando resposta')

        resposta = chain.invoke({'question': nova_mensagem})
        st.session_state['ultima_resposta'] = resposta
        st.rerun()

def main():
    with st.sidebar:
        sidebar()
        inform()
    chat_window()

if __name__ == '__main__':
    main()
