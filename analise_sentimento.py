import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# importando pdf pelo path
from pdfminer.high_level import extract_text
#pip install pdfminer.six
# NAO ESQUECER DE MUDAR O ARQUIVO
pdf_path = '/home/gab/Desktop/paitom/art60.pdf'
texto_pdf = extract_text(pdf_path)

# todos os caracteres em minusculo
texto_processado = texto_pdf.lower()

# NAO ESQUECER DE MUDAR O IDIOMA, OPCOES: 'english' 'spanish' 'portuguese
stop_words = set(stopwords.words('english'))
texto_processado = ' '.join(word for word in texto_processado.split() if word not in stop_words)

# removendo links
import re 
def remover_links(texto):
    # Expressão regular para identificar links
    link_regex = r'(https?:\/\/)(www\.)?([\w\-\.]+)([/\?\=\&\%\.]+)?([^\s\<\>\'\'\[\]\{\}\(\)]+)*'
    return re.sub(link_regex, ' ', texto)
texto_sem_links = remover_links(texto_processado)

# texto apenas com letras: 
def manter_somente_letras(texto):
    palavras = []
    for palavra in texto.split():
        palavra_sem_caracteres = ''.join(char for char in palavra if char.isalpha())
        if palavra_sem_caracteres:
            palavras.append(palavra_sem_caracteres)
    return ' '.join(palavras)

texto_letras = manter_somente_letras(texto_sem_links)

# lematizando
lemmatizer = WordNetLemmatizer()
texto_final = ' '.join(lemmatizer.lemmatize(word) for word in texto_letras.split())
texto_lista = texto_final.split()

#contagem das palavras para keyword cloud
from collections import Counter
palavras_freq = Counter(texto_final.split())
palavras_mais_frequentes = palavras_freq.most_common(50)

#criando um dicionario para a wordcloud com as 50 palavras mais frequentes
palavras_top_50_freq = dict(palavras_mais_frequentes)
apenaspalavras = []
for palavra, numero in palavras_top_50_freq.items():
    apenaspalavras.extend([palavra]*numero)    

from wordcloud import WordCloud
nuvem = WordCloud( background_color='white', max_words=50).generate_from_frequencies(palavras_top_50_freq)

#salvando como png
nuvem.to_file('nuvem_palavras_top_50.png')

# agora com o texto tratado, comecamos a analise:
sentimento = SentimentIntensityAnalyzer()
scores_sentimento = []

# fazendo a analise das frases
for frase in texto_final.split('.'):
    scores_sentimento.append(sentimento.polarity_scores(frase))

# gerando o sentimento e printando o valor com o resultado
sentimento_geral = sum(score['compound'] for score in scores_sentimento) / len(scores_sentimento)
if sentimento_geral > 0.05:
    feedback_geral = 'Positivo/otimista'
elif sentimento_geral < -0.05:
    feedback_geral = 'Negativo/pessimista'
else:
    feedback_geral = 'Neutro'

# analisando as emocoes pelo VADER
# pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
sentiment = analyzer.polarity_scores(texto_pdf)
sentiment_keywords = analyzer.polarity_scores(palavras_top_50_freq)


# pip install TextBlob
from textblob import TextBlob
from collections import Counter

def identificar_emocao(texto, idioma):

  # dicionario
  emocoes = {
    'pt': {
      # emocoes positivas
      'apoio': ['apoiar', 'concordar', 'aprovar', 'endossar', 'auxiliar', 'sustentar', 'favorecer', 'incentivar', 'reforçar', 'solidarizar'],
      'confiança': ['confiar', 'seguro', 'otimista', 'esperançoso', 'fé', 'certeza', 'tranquilidade', 'convicção', 'credo'],
      'cooperação': ['cooperar', 'colaborar', 'unir', 'sinergia', 'companheirismo', 'entreajuda', 'mútua ajuda', 'colaboração', 'conjugação de esforços'],
      'eficiência': ['eficiente', 'produtivo', 'rentável', 'otimizar', 'eficaz', 'desempenho', 'rapidez', 'precisão', 'habilidade'],
      'empatia': ['compreender', 'compaixão', 'considerar', 'solicitude', 'identificação', 'sintonia', 'escuta ativa', 'sensibilidade', 'solidariedade'],
      'equidade': ['justo', 'igual', 'imparcial', 'isonomia', 'justiça', 'equilíbrio', 'igualdade', 'proporcionalidade', 'equanimidade'],
      'liberdade': ['livre', 'autônomo', 'independente', 'autodeterminação', 'emancipação', 'autonomia', 'libertação', 'soberania', 'franquia'],
      'progresso': ['avançar', 'desenvolver', 'crescer', 'aprimorar', 'evolução', 'melhoria', 'progredir', 'ascensão', 'desenvolvimento'],
      'segurança': ['seguro', 'estável', 'tranquilo', 'confiável', 'proteção', 'garantia', 'resguardo', 'blindagem', 'salvaguarda'],
      'transparência': ['claro', 'aberto', 'honesto', 'accountability', 'transparência', 'visibilidade', 'sinceridade', 'legibilidade', 'franqueza'],
      
      # emocoes negativas
      'apatia': ['apático', 'desanimado', 'desinteressado', 'letárgico', 'indiferente', 'inerte', 'passivo', 'desmotivado', 'langoroso'],
      'autocracia': ['autocrático', 'autoritário', 'tirano', 'déspota', 'totalitarismo', 'ditadura', 'absolutismo', 'centralização', 'despotismo'],
      'censura': ['censurar', 'restringir', 'controlar', 'oprimir', 'repressão', 'proibição', 'silêncio', 'limitação', 'supressão'],
      'corrupção': ['corrupto', 'fraudulento', 'venal', 'nepotismo', 'desonestidade', 'suborno', 'propina', 'peculato', 'prevaricação'],
      'desigualdade': ['injusto', 'desigual', 'discrepante', 'assimetria', 'desequilíbrio', 'iniquidade', 'disparidade', 'desproporção', 'hiato'],
      'discriminação': ['preconceituoso', 'intolerante', 'segregar', 'marginalizar', 'racismo', 'sexismo', 'xenofobia', 'homofobia', 'discriminação'],
      'fanatismo': ['fanático', 'radical', 'extremista', 'intolerante', 'obcecado', 'ceticismo', 'dogmatismo', 'fundamentalismo', 'sectarismo'],
      'ineficiência': ['ineficiente', 'improdutivo', 'desperdício', 'incompetente', 'ineficácia', 'lentidão', 'preguiça', 'inabilidade', 'imprevidência'],
      'insegurança': ['inseguro', 'vulnerável', 'medo', 'apreensão', 'ansiedade', 'timidez', 'desconfiança', 'incerteza', 'inquietude'],
      'instabilidade': ['instável', 'volátil', 'caótico', 'desordem', 'inconstância', 'imprevisibilidade', 'turbulência', 'agitação', 'labilidade'],
      'mentira': ['mentiroso', 'falso', 'desinformação', 'propaganda', 'engano', 'embuste', 'farsa', 'manipulação', 'falácia'],
      'opressão': ['oprimir', 'reprimir', 'tirano', 'cruel', 'subjugação', 'dominação', 'sujeição', 'escravidão', 'subjugamento'],
      'polarização': ['polarizado', 'fragmentado', 'sectário', 'antagonismo', 'divisão', 'radicalização', 'conflito', 'oposição', 'bifurcação'],
      'violência': ['violento', 'agressivo', 'cruel', 'hostil', 'brutalidade', 'truculência', 'beligerância', 'combate', 'barbárie'],
      
      # emocoes neutras
      'análise': ['analisar', 'investigar', 'estudar', 'examinar', 'dissecação', 'interpretação', 'avaliação', 'scrutínio', 'diagnóstico'],
      'avaliação': ['avaliar', 'julgar', 'ponderar', 'mensurar', 'exame', 'critério', 'veredicto', 'estimativa', 'apreciação'],
      'comparação': ['comparar', 'cotejar', 'paralelo', 'analogia', 'contraste', 'semelhança', 'confronto', 'oposição', 'justaposição'],
      'descrição': ['descrever', 'narrar', 'expor', 'caracterizar', 'relatório', 'representação', 'delineamento', 'detalhamento', 'depicção'],
      'explicação': ['explicar', 'esclarecer', 'justificar', 'fundamentar', 'interpretação', ' elucidação', 'demonstração', 'argumentação', 'exposição'],
      'neutralidade': ['neutro', 'imparcial', 'equidistante', 'objetividade', 'isenção', 'desapaixonamento', 'indiferença', 'neutralização', 'neutralidade'],
    },
    'en': {
      # emocoes positivas
      'support': ['support', 'agree', 'approve', 'endorse', 'assist', 'sustain', 'favor', 'encourage', 'reinforce', 'solidarity'],
      'confidence': ['trust', 'confident', 'optimistic', 'hopeful', 'faith', 'certainty', 'peace of mind', 'conviction', 'creed'],
      'cooperation': ['cooperate', 'collaborate', 'unite', 'synergy', 'companionship', 'mutual aid', 'collaboration', 'joint effort'],
      'efficiency': ['efficient', 'productive', 'profitable', 'optimize', 'effective', 'performance', 'speed', 'accuracy', 'skill'],
      'empathy': ['understand', 'compassion', 'consider', 'solicitude', 'identification', 'attunement', 'active listening', 'sensitivity', 'solidarity'],
      'equity': ['fair', 'equal', 'impartial', 'isonomy', 'justice', 'balance', 'equality', 'proportionality', 'equanimity'],
      'freedom': ['free', 'autonomous', 'independent', 'self-determination', 'emancipation', 'autonomy', 'liberation', 'sovereignty', 'entitlement'],
      'progress': ['advance', 'develop', 'grow', 'improve', 'evolution', 'enhancement', 'progress', 'rise', 'development'],
      'security': ['safe', 'stable', 'calm', 'reliable', 'protection', 'assurance', 'safeguard', 'shielding', 'protection'],
      'transparency': ['clear', 'open', 'honest', 'accountability', 'transparency', 'visibility', 'sincerity', 'legibility', 'candor'],

      # emocoes negativas
      'apathy': ['apathetic', 'discouraged', 'uninterested', 'lethargic', 'indifferent', 'inert', 'passive', 'unmotivated', 'languid'],
      'autocracy': ['autocratic', 'authoritarian', 'tyrant', 'despot', 'totalitarianism', 'dictatorship', 'absolutism', 'centralization', 'despotism'],
      'censorship': ['censor', 'restrict', 'control', 'oppress', 'repression', 'prohibition', 'silence', 'limitation', 'suppression'],
      'corruption': ['corrupt', 'fraudulent', 'venal', 'nepotism', 'dishonesty', 'bribery', 'kickback', 'embezzlement', 'prevarication'],
      'inequality': ['unfair', 'unequal', 'discrepant', 'asymmetry', 'imbalance', 'iniquity', 'disparity', 'disproportion', 'gap'],
      'discrimination': ['prejudiced', 'intolerant', 'segregate', 'marginalize', 'racism', 'sexism', 'xenophobia', 'homophobia', 'discrimination'],
      'fanaticism': ['fanatic', 'radical', 'extremist', 'intolerant', 'obsessed', 'skepticism', 'dogmatism', 'fundamentalism', 'sectarianism'],
      'inefficiency': ['inefficient', 'unproductive', 'waste', 'incompetent', 'ineffectiveness', 'slowness', 'laziness', 'inability', 'improvidence'],
      'insecurity': ['insecure', 'vulnerable', 'fear', 'apprehension', 'anxiety', 'shyness', 'distrust', 'uncertainty', 'unease'],
      'instability': ['unstable', 'volatile', 'chaotic', 'disorder', 'inconstancy', 'unpredictability', 'turbulence', 'agitation', 'lability'],
      'lie': ['liar', 'false', 'misinformation', 'propaganda', 'deceit', 'trickery', 'farce', 'manipulation', 'fallacy'],
      'oppression': ['oppress', 'repress', 'tyrant', 'cruel', 'subjugation', 'domination', 'subjection', 'slavery', 'subjugation'],
      'polarization': ['polarized', 'fragmented', 'sectarian', 'antagonism', 'division', 'radicalization', 'conflict', 'opposition', 'bifurcation'],
      'violence': ['violent', 'aggressive', 'cruel', 'hostile', 'brutality', 'truculence', 'belligerence', 'combat', 'barbarity'],
      
      # emocoes neutras
      'analysis': ['analyze', 'investigate', 'study', 'examine', 'dissection', 'interpretation', 'assessment', 'scrutiny', 'diagnosis'],
      'evaluation': ['evaluate', 'judge', 'ponder', 'measure', 'examination', 'criterion', 'verdict', 'estimate', 'appreciation'],
      'comparison': ['compare', 'contrast', 'parallel', 'analogy', 'contrast', 'similarity', 'confrontation', 'opposition', 'juxtaposition'],
      'description': ['describe', 'narrate', 'explain', 'characterize', 'report', 'representation', 'outline', 'detailing', 'depiction'],
      'explanation': ['explain', 'clarify', 'justify', 'substantiate', 'interpretation', 'elucidation', 'demonstration', 'argumentation', 'exposition'],
      'neutrality': ['neutral', 'impartial', 'equidistant', 'objectivity', 'exemption', 'dispassion', 'indifference', 'neutralization', 'neutrality'],

    },
    'es': {
      # emocoes positivas
      'apoyo': ['apoyar', 'concordar', 'aprobar', 'aprobar', 'auxiliar', 'sostener', 'favorecer', 'incentivar', 'reforzar', 'solidarizarse'],
      'confianza': ['confiar', 'seguro', 'optimista', 'esperanzado', 'fe', 'certeza', 'tranquilidad', 'convicción', 'credo'],
      'cooperación': ['cooperar', 'colaborar', 'unir', 'sinergia', 'compañerismo', 'entreayuda', 'ayuda mutua', 'colaboración', 'conjunción de esfuerzos'],
      'eficiencia': ['eficiente', 'productivo', 'rentable', 'optimizar', 'eficaz', 'rendimiento', 'rapidez', 'precisión', 'habilidad'],
      'empatía': ['comprender', 'compasión', 'considerar', 'solicitud', 'identificación', 'sintonía', 'escucha activa', 'sensibilidad', 'solidaridad'],
      'equidad': ['justo', 'igual', 'imparcial', 'isonomía', 'justicia', 'equilibrio', 'igualdad', 'proporcionalidad', 'ecuanimidad'],
      'libertad': ['libre', 'autónomo', 'independiente', 'autodeterminación', 'emancipación', 'autonomía', 'liberación', 'soberanía', 'franquicia'],
      'progreso': ['avanzar', 'desarrollar', 'crecer', 'mejorar', 'evolución', 'mejora', 'progresar', 'ascenso', 'desarrollo'],
      'seguridad': ['seguro', 'estable', 'tranquilo', 'confiable', 'protección', 'garantía', 'resguardo', 'blindaje', 'salvaguarda'],
      'transparencia': ['claro', 'abierto', 'honesto', 'accountability', 'transparencia', 'visibilidad', 'sinceridad', 'legibilidad', 'franqueza'],

      
      # emocoes negativas
      'apatía': ['apático', 'desanimado', 'desinteresado', 'letárgico', 'indiferente', 'inerte', 'pasivo', 'desmotivado', 'langoroso'],
      'autocracia': ['autocrático', 'autoritario', 'tirano', 'déspota', 'totalitarismo', 'dictadura', 'absolutismo', 'centralización', 'despotismo'],
      'censura': ['censurar', 'restringir', 'controlar', 'oprimir', 'represión', 'prohibición', 'silencio', 'limitación', 'supresión'],
      'corrupción': ['corrupto', 'fraudulento', 'venal', 'nepotismo', 'deshonestidad', 'soborno', 'propina', 'peculado', 'prevaricación'],
      'desigualdad': ['injusto', 'desigual', 'discrepante', 'asimetría', 'desequilibrio', 'iniquidad', 'disparidad', 'desproporción', 'hiato'],
      'discriminación': ['prejuicioso', 'intolerante', 'segregar', 'marginar', 'racismo', 'sexismo', 'xenofobia', 'homofobia', 'discriminación'],
      'fanatismo': ['fanático', 'radical', 'extremista', 'intolerante', 'obsesionado', 'ceticismo', 'dogmatismo', 'fundamentalismo', 'sectarismo'],
      'ineficiencia': ['ineficiente', 'improductivo', 'desperdicio', 'incompetente', 'ineficacia', 'lentitud', 'pereza', 'inhabilidad', 'imprevisión'],
      'inseguridad': ['inseguro', 'vulnerable', 'miedo', 'aprensión', 'ansiedad', 'timidez', 'desconfianza', 'incertidumbre', 'inquietud'],
      'inestabilidad': ['inestable', 'volátil', 'caótico', 'desorden', 'inconstancia', 'imprevisibilidad', 'turbulencia', 'agitación', 'labilidad'],
      'mentira': ['mentiroso', 'falso', 'desinformación', 'propaganda', 'engaño', 'embuste', 'farsa', 'manipulación', 'falacia'],
      'opresión': ['oprimir', 'reprimir', 'tirano', 'cruel', 'subyugación', 'dominación', 'sujeción', 'esclavitud', 'subyugamiento'],
      'polarización': ['polarizado', 'fragmentado', 'sectario', 'antagonismo', 'división', 'radicalización', 'conflicto', 'oposición', 'bifurcación'],
      'violencia': ['violento', 'agresivo', 'cruel', 'hostil', 'brutalidad', 'truculencia', 'beligerancia', 'combate', 'barbarie'],

      
      # emocoes neutras
      'análisis': ['analizar', 'investigar', 'estudiar', 'examinar', 'disección', 'interpretación', 'evaluación', 'escrutinio', 'diagnóstico'],
      'evaluación': ['evaluar', 'juzgar', 'ponderar', 'mensurar', 'examen', 'criterio', 'veredicto', 'estimación', 'apreciación'],
      'comparación': ['comparar', 'cotejar', 'paralelo', 'analogía', 'contraste', 'semejanza', 'confrontación', 'oposición', 'yuxtaposición'],
      'descripción': ['describir', 'narrar', 'exponer', 'caracterizar', 'relato', 'representación', 'delineación', 'detallado', 'descripción'],
      'explicación': ['explicar', 'aclarar', 'justificar', 'fundamentar', 'interpretación', 'elucidación', 'demostración', 'argumentación', 'exposición'],
      'neutralidad': ['neutro', 'imparcial', 'equidistante', 'objetividad', 'exención', 'desapasionamiento', 'indiferencia', 'neutralización', 'neutralidad'],

    },
  }

  # Contagem de palavras por emoção
  contagem_emocoes = Counter()
  for palavra in texto:
    for emocao, palavras_emocao in emocoes[idioma].items():
      if palavra in palavras_emocao:
        contagem_emocoes[emocao] += 1

  # Emoção predominante
  emocao_predominante = None
  maior_contagem = 0
  for emocao, contagem in contagem_emocoes.items():
    if contagem > maior_contagem:
      maior_contagem = contagem
      emocao_predominante = emocao
    return emocao_predominante

# NAO ESQUECER DE MUDAR O IDIOMA, OPCOES: 'en' 'pt' 'es'
emocao_especifica_texto = identificar_emocao(texto_lista,'en')
emocao_especifica_keywords = identificar_emocao(apenaspalavras,'en')

# NAO ESQUECER DE MUDAR O NOMEDOARQUIVO.txt
with open('ANALISESENTIMENTO.txt', 'w') as arquivo:
    arquivo.write('As 50 palavras mais frequentes são:\n')
    arquivo.write(str(palavras_top_50_freq) + '\n\n')
    arquivo.write('Com um valor composto de: {} o feedback geral do artigo é: {}\n\n'.format(sentimento_geral, feedback_geral))
    arquivo.write('O VADER sentiment analyzer indica que o texto é:\n')
    arquivo.write('Positivo/otimista: {}\n'.format(sentiment['pos']))
    arquivo.write('Negativo/pessimista: {}\n'.format(sentiment['neg']))
    arquivo.write('Neutro: {}\n'.format(sentiment['neu']))
    arquivo.write('Composto: {}\n\n'.format(sentiment['compound']))
    arquivo.write('O VADER sentiment analyzer indica que as keywords do texto são:\n')
    arquivo.write('Positivo/otimista: {}\n'.format(sentiment_keywords['pos']))
    arquivo.write('Negativo/pessimista: {}\n'.format(sentiment_keywords['neg']))
    arquivo.write('Neutro: {}\n\n'.format(sentiment_keywords['neu']))
    arquivo.write('A emocao especifica do texto pelo dicionário indica: {}\n'.format(emocao_especifica_texto))
    arquivo.write('A emocao especifica das keywords pelo dicionário indica: {}\n'.format(emocao_especifica_keywords))

with open('WORDCLOUD.txt', 'w') as arquivo:
    arquivo.write(str(apenaspalavras) + '\n\n')