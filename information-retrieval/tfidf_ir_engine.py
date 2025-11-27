# Paulo Daniel Forti da Fonseca - 12311BSI321
# Pedro Henrique Lopes Duarte - 12311BSI237

import json
import unicodedata
import re
import math

#Caminho para o arquivo JSON com a coleção bruta 
class SistemaRI:
    def __init__(self, caminho_json):
        # Caminho para o arquivo JSON com a coleção bruta
        self.caminho_json = caminho_json

        # Lista de documentos brutos lidos do JSON (ainda não inseridos na coleção)
        self.documentos_json = []  # cada item pode ser {"id": ..., "texto": ...} ou algo similar
        self.proximo_idx_json = 0  # controla qual é o próximo doc a ser adicionado

        # Coleção atual (documentos inseridos no sistema)
        # cada doc: {"id": int, "texto": str, "tokens": list[str]}
        self.documentos = []

        # Estruturas de RI
        self.vocabulario = []          # lista de termos
        self.indice_invertido = {}     # { termo: { doc_id: [pos1, pos2, ...], ... } }

        # Estruturas TF-IDF
        self.tf = []       # lista de vetores/dicts por documento
        self.idf = {}      # { termo: valor_idf }
        self.tfidf = []    # lista de vetores/dicts por documento

        # Carregar o JSON assim que o sistema é criado
        self.carregar_json()

    # --------- JSON E COLEÇÃO ----------- 
    
    def carregar_json(self):
        try:
            with open(self.caminho_json, 'r', encoding='utf-8') as f:
                dados = json.load(f) # espera-se uma lista de documentos
        except FileNotFoundError:
            print(f"Arquivo JSON não encontrado: {self.caminho_json}")
            dados = []

        self.documentos_json = dados
        self.proximo_idx_json = 0
        print(f"{len(self.documentos_json)} documentos carregados do JSON.")    

    def adicionar_proximo_documento(self):
        #Adiciona um documento por vez, seguindo a ordem do JSON.
        if self.proximo_idx_json >= len(self.documentos_json):
            print("Não há mais documentos no JSON para adicionar.")
            return

        doc_bruto = self.documentos_json[self.proximo_idx_json]
        self.proximo_idx_json += 1

        # JSON: campos 'name' e 'content'
        doc_id = doc_bruto.get("name", f"D{self.proximo_idx_json}")
        texto = doc_bruto.get("content", "")

        tokens = self.limpar_texto(texto)

        doc = {
            "id": doc_id,
            "texto": texto,
            "tokens": tokens
        }
        self.documentos.append(doc)
        print(f"Documento {doc_id} adicionado à coleção.")

        # Atualizar estruturas
        self.reconstruir_estruturas()

    def adicionar_todos_documentos(self):
        #Adiciona todos os documentos do JSON à coleção de uma vez.
        while self.proximo_idx_json < len(self.documentos_json):
            self.adicionar_proximo_documento()

    def remover_documento_por_id(self, doc_id):
        #Remove um documento da coleção pelo seu identificador (ex: 'D1').
        antes = len(self.documentos)
        self.documentos = [doc for doc in self.documentos if doc["id"] != doc_id]
        depois = len(self.documentos)

        if antes == depois:
            print(f"Nenhum documento com id {doc_id} foi encontrado.")
        else:
            print(f"Documento {doc_id} removido da coleção.")
            # Atualizar estruturas
            self.reconstruir_estruturas()

    # --------- UTILITÁRIOS DE TEXTO -----------    
            
    def remover_acentos(self, texto):
        nfkd = unicodedata.normalize("NFKD", texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)])

    def limpar_texto(self, texto):
        # minúsculas
        texto = texto.lower()
        # remover acentos
        texto = self.remover_acentos(texto)
        # substituir qualquer coisa que não seja letra ou número por espaço
        texto = re.sub(r"[^a-z0-9]+", " ", texto)
        # quebrar em tokens
        tokens = texto.split()
        # AQUI depois vamos remover stopwords e aplicar radicalização
        return tokens

# ---------- ESTRUTURAS DE ÍNDICE ----------

    def reconstruir_estruturas(self):
        self.reconstruir_vocabulario()
        self.reconstruir_indice_invertido()
        self.reconstruir_tf_idf()

    def reconstruir_vocabulario(self):
        termos = set()
        for doc in self.documentos:
            for t in doc["tokens"]:
                termos.add(t)
        self.vocabulario = sorted(list(termos))

    def reconstruir_indice_invertido(self):
        # { termo: { doc_id: [pos1, pos2, ...] } }
        indice = {}

        for doc in self.documentos:
            doc_id = doc["id"]
            tokens = doc["tokens"]
            for pos, termo in enumerate(tokens):
                if termo not in indice:
                    indice[termo] = {}
                if doc_id not in indice[termo]:
                    indice[termo][doc_id] = []
                indice[termo][doc_id].append(pos)

        self.indice_invertido = indice

    def reconstruir_tf_idf(self):
        """Recalcula TF, IDF e TF-IDF para a coleção atual."""
        # zera estruturas
        self.tf = []
        self.idf = {}
        self.tfidf = []

        # se não há documentos ou vocabulário, não faz nada
        if not self.documentos or not self.vocabulario:
            return

        # ---------- TF (log) ----------
        # para cada documento, cria um vetor/dict com tf_log2 por termo do vocabulário
        lista_tf = []
        for doc in self.documentos:
            tokens = doc["tokens"]
            contagem = {}

            # contagem bruta de termos no documento
            for t in tokens:
                contagem[t] = contagem.get(t, 0) + 1

            # tf log2: 1 + log2(freq) se freq > 0, senão 0
            tf_doc = {}
            for termo in self.vocabulario:
                freq = contagem.get(termo, 0)
                if freq > 0:
                    tf_doc[termo] = 1 + math.log2(freq)
                else:
                    tf_doc[termo] = 0.0
            lista_tf.append(tf_doc)

        self.tf = lista_tf

        # ---------- IDF ----------
        # df(t) = em quantos documentos o termo aparece
        N = len(self.documentos)
        df = {termo: 0 for termo in self.vocabulario}

        for termo in self.vocabulario:
            for doc_tf in self.tf:
                if doc_tf[termo] > 0:
                    df[termo] += 1

        idf = {}
        for termo in self.vocabulario:
            if df[termo] > 0:
                idf[termo] = math.log2(N / df[termo])
            else:
                idf[termo] = 0.0

        self.idf = idf

        # ---------- TF-IDF ----------
        lista_tfidf = []
        for doc_tf in self.tf:
            tfidf_doc = {}
            for termo in self.vocabulario:
                tfidf_doc[termo] = doc_tf[termo] * self.idf[termo]
            lista_tfidf.append(tfidf_doc)

        self.tfidf = lista_tfidf

    # ---------- MÉTODOS DE EXIBIÇÃO ----------

    def exibir_vocabulario(self):
        print("Vocabulário da coleção:")
        for termo in self.vocabulario:
            print(termo)
        print(f"Total de termos distintos: {len(self.vocabulario)}")

    def exibir_indice_invertido(self):
        print("Índice invertido (termo -> {doc_id: [posições]}):")
        for termo, postings in self.indice_invertido.items():
            print(f"{termo}: {postings}")

    def exibir_tfidf(self):
        if not self.tfidf:
            print("Matriz TF-IDF ainda não calculada (talvez não haja documentos).")
            return
        
        if  len(self.documentos) < 2:
            print("Não é possível criar a matriz TF-IDF com apenas um documento na coleção.")
            return

        print("Matriz TF-IDF (valores diferentes de zero):")
        for doc, vetor in zip(self.documentos, self.tfidf):
            print(f"\nDocumento {doc['id']}:")
            # mostra só termos com tf-idf > 0
            for termo, valor in vetor.items():
                if valor > 0:
                    print(f"  {termo}: {valor:.4f}")

    # ---------- CONSULTA BOOLEANA ----------

    def _docs_para_termo(self, termo_normalizado):
        """Retorna o conjunto de IDs de documentos que contêm o termo."""
        postings = self.indice_invertido.get(termo_normalizado, {})
        return set(postings.keys())

    def consulta_booleana(self, consulta_str):
        """
        Consulta booleana simples com operadores AND, OR, NOT.
        Exemplo: 'brasil AND liberdade', 'brasil OR povo', 'brasil AND NOT colonia'
        Avaliação da esquerda para a direita (sem precedência sofisticada).
        """
        if not self.documentos:
            print("Não há documentos na coleção.")
            return

        # conjunto com TODOS os docs (para usar em NOT)
        todos_docs = set(doc["id"] for doc in self.documentos)

        # separar por espaço
        tokens = consulta_str.strip().split()

        if not tokens:
            print("Consulta vazia.")
            return

        # vamos manter 2 variáveis:
        #   resultado_atual: conjunto de docs até agora
        #   operador_atual: AND / OR (entre blocos)
        resultado_atual = None
        operador_atual = "AND"  # por padrão, se não especificar, assumimos AND

        i = 0
        while i < len(tokens):
            token = tokens[i].upper()

            if token in ("AND", "OR"):
                operador_atual = token
                i += 1
                continue

            negacao = False
            if token == "NOT":
                negacao = True
                i += 1
                if i >= len(tokens):
                    print("Erro na consulta: NOT no final sem termo.")
                    return
                termo_bruto = tokens[i]
            else:
                termo_bruto = tokens[i]

            # normalizar o termo da mesma forma que os documentos
            termos_norm = self.limpar_texto(termo_bruto)
            if not termos_norm:
                # termo virou vazio depois da limpeza
                conjunto_termo = set()
            else:
                termo_norm = termos_norm[0]
                conjunto_termo = self._docs_para_termo(termo_norm)

            if negacao:
                conjunto_termo = todos_docs - conjunto_termo

            # combinar com resultado_atual usando operador_atual
            if resultado_atual is None:
                resultado_atual = conjunto_termo
            else:
                if operador_atual == "AND":
                    resultado_atual = resultado_atual & conjunto_termo
                elif operador_atual == "OR":
                    resultado_atual = resultado_atual | conjunto_termo

            i += 1

        if not resultado_atual:
            print("Nenhum documento satisfaz a consulta booleana.")
        else:
            print("Documentos que satisfazem a consulta booleana:")
            for doc_id in sorted(resultado_atual):
                print(" ", doc_id)

    # ---------- CONSULTA POR SIMILARIDADE (COSSENO) ----------

    def _vetor_tfidf_para_texto(self, texto):
        """
        Dado um texto de consulta, produz um vetor TF-IDF (dict termo -> peso)
        usando o vocabulário e o IDF atuais.
        """
        if not self.vocabulario or not self.idf:
            return {}

        tokens = self.limpar_texto(texto)
        contagem = {}
        for t in tokens:
            contagem[t] = contagem.get(t, 0) + 1

        # TF log2
        tf_q = {}
        for termo in self.vocabulario:
            freq = contagem.get(termo, 0)
            if freq > 0:
                tf_q[termo] = 1 + math.log2(freq)
            else:
                tf_q[termo] = 0.0

        # TF-IDF
        tfidf_q = {}
        for termo in self.vocabulario:
            tfidf_q[termo] = tf_q[termo] * self.idf.get(termo, 0.0)

        return tfidf_q

    def _cosseno(self, v1, v2):
        """
        Calcula similaridade de cosseno entre dois vetores representados
        como dicionários termo -> peso.
        """
        # produto escalar
        dot = 0.0
        for termo, peso1 in v1.items():
            peso2 = v2.get(termo, 0.0)
            dot += peso1 * peso2

        # norma de v1
        norm1 = math.sqrt(sum(peso**2 for peso in v1.values()))
        # norma de v2
        norm2 = math.sqrt(sum(peso**2 for peso in v2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def consulta_similaridade(self, texto_consulta, k=10):
        """
        Calcula similaridade de cosseno entre a consulta e todos os documentos,
        usando os vetores TF-IDF.
        Mostra os top-k documentos mais similares.
        """
        if not self.documentos or not self.tfidf:
            print("Coleção ou TF-IDF vazios. Adicione documentos primeiro.")
            return

        vetor_q = self._vetor_tfidf_para_texto(texto_consulta)
        if not vetor_q:
            print("Vetor de consulta vazio (talvez vocabulário ou IDF vazios?).")
            return

        resultados = []
        for doc, vetor_d in zip(self.documentos, self.tfidf):
            sim = self._cosseno(vetor_q, vetor_d)
            resultados.append((doc["id"], sim))

        # ordenar por similaridade decrescente
        resultados.sort(key=lambda x: x[1], reverse=True)

        print("Ranking por similaridade de cosseno:")
        for doc_id, sim in resultados[:k]:
            print(f"  Doc {doc_id}: similaridade = {sim:.4f}")

    # ---------- CONSULTA POR FRASE ----------

    def consulta_frase(self, frase_str):
        """
        Busca documentos que contenham a frase exata (em termos normalizados),
        usando o índice invertido com posições.
        """
        if not self.documentos:
            print("Não há documentos na coleção.")
            return

        termos = self.limpar_texto(frase_str)
        if not termos:
            print("Frase vazia após normalização.")
            return

        # se só tiver um termo, vira uma consulta simples: presença do termo
        if len(termos) == 1:
            termo = termos[0]
            docs = self.indice_invertido.get(termo, {})
            if not docs:
                print("Nenhum documento contém o termo da frase.")
                return
            print("Documentos que contêm o termo:")
            for doc_id in docs.keys():
                print(" ", doc_id)
            return

        # termos[0], termos[1], ..., termos[n-1]
        # primeiro, encontrar docs que têm TODOS os termos
        candidatos = None
        for termo in termos:
            postings = self.indice_invertido.get(termo, {})
            docs_termo = set(postings.keys())
            if candidatos is None:
                candidatos = docs_termo
            else:
                candidatos = candidatos & docs_termo

        if not candidatos:
            print("Nenhum documento contém todos os termos da frase.")
            return

        # agora, para cada doc candidato, verificar se há sequência de posições
        resultados = []  # (doc_id, num_ocorrencias)

        for doc_id in candidatos:
            # lista de listas de posições por termo
            listas_pos = []
            for termo in termos:
                posicoes = self.indice_invertido[termo][doc_id]
                listas_pos.append(posicoes)

            # queremos contar quantas vezes existe p tal que:
            # p em listas_pos[0], p+1 em listas_pos[1], ..., p+(n-1) em listas_pos[n-1]
            ocorrencias = 0
            primeiras_pos = listas_pos[0]
            n = len(termos)

            for p in primeiras_pos:
                encontrado = True
                for i in range(1, n):
                    pos_esperada = p + i
                    if pos_esperada not in listas_pos[i]:
                        encontrado = False
                        break
                if encontrado:
                    ocorrencias += 1

            if ocorrencias > 0:
                resultados.append((doc_id, ocorrencias))

        if not resultados:
            print("Nenhum documento contém a frase exata.")
            return

        # ordenar por número de ocorrências (maior primeiro)
        resultados.sort(key=lambda x: x[1], reverse=True)

        print("Documentos que contêm a frase (ordenados por número de ocorrências):")
        for doc_id, qtd in resultados:
            print(f"  Doc {doc_id}: {qtd} ocorrência(s)")


# ---------- MENU PRINCIPAL ----------

def main():
    caminho_json = r"collection_tfidf_ir_engine.json"   # ajuste se o nome for diferente
    sistema = SistemaRI(caminho_json)

    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1) Adicionar um documento da lista (JSON)")
        print("2) Adicionar todos os documentos da lista (JSON)")
        print("3) Remover um documento da coleção pelo ID (ex: D1)")
        print("4) Exibir vocabulário")
        print("5) Exibir matriz TF-IDF")
        print("6) Exibir índice invertido completo por posição")
        print("7) Realizar consulta booleana (TODO)")
        print("8) Realizar consulta por similaridade (TODO)")
        print("9) Realizar consulta por frase (TODO)")
        print("10) Sair")

        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            sistema.adicionar_proximo_documento()
        elif opcao == "2":
            sistema.adicionar_todos_documentos()
        elif opcao == "3":
            doc_id = input("Informe o ID do documento a remover (ex: D1): ").strip()
            sistema.remover_documento_por_id(doc_id)
        elif opcao == "4":
            sistema.exibir_vocabulario()
        elif opcao == "5":
            sistema.exibir_tfidf()
        elif opcao == "6":
            sistema.exibir_indice_invertido()
        elif opcao == "7":
            consulta = input("Digite a consulta booleana (ex: brasil AND liberdade): ")
            sistema.consulta_booleana(consulta)
        elif opcao == "8":
            texto = input("Digite o texto da consulta para similaridade: ")
            sistema.consulta_similaridade(texto)
        elif opcao == "9":
            frase = input("Digite a frase a ser buscada: ")
            sistema.consulta_frase(frase)
        elif opcao == "10":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
