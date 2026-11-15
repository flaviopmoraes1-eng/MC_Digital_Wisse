# PIPE-FEA

App Streamlit para cálculos de engenharia.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No Streamlit Community Cloud, mantenha `packages.txt` na mesma pasta do
`app.py`. Ele instala a biblioteca de sistema necessária pelo Gmsh.

## Funcionalidades

- Upload de geometria `.step`/`.stp` (casca/superfícies) ou `.stl` (malha triangular; usada como está e refinável)
- Malha triangular gerada com Gmsh (`mesher.py`), com controle de densidade e refinamento
- Visualização 3D interativa (three.js embutido em `vendor/`, sem internet): girar, mover e zoom
- Seleção com o mouse de arestas e superfícies (clique; Shift+clique soma à seleção)
- Arestas: ancoragem (6 GDL = 0) ou restrição personalizada de UX, UY, UZ, RX, RY, RZ com valor imposto
- Superfícies: pressão (positiva = comprime a casca, contrária à normal)
- As condições ficam em `st.session_state.bc` (`curves`: tag → {GDL: valor}, `pressure`: tag → valor)
- Análise linear elástica de casca (`fea.py`): triângulo plano com 6 GDL/nó — membrana CST, flexão/cisalhamento Reissner-Mindlin (DSG3, sem travamento) e rigidez fictícia de perfuração
- Antes de calcular, verifica o equilíbrio estático: as cargas não podem realizar trabalho nos movimentos de corpo rígido que os apoios deixam livres, e os apoios devem restringir os 6 movimentos. Após resolver, confere cargas + reações = 0 (força e momento)
- Resultados: campo em lista suspensa acima da imagem (Von Mises em MPa, deformação equivalente em %, deslocamento resultante e componentes UX/UY/UZ em mm) e posição na espessura também em lista suspensa acima da imagem: PL (Tensão de Membrana), PL+PB_bottom (Tensão de Membrana + Flexão, base) e PL+PB_top (Tensão de Membrana + Flexão, topo). Reações e configuração deformada, com mapa de cores no modelo 3D; ao passar o mouse sobre um nó aparece o valor dele e uma bandeirinha vermelha "Máx." marca o nó de maior valor. A deformação equivalente é a elástica (εeq = σ_VM/E, em %) na mesma posição da espessura
- Pré-processamento da geometria: arestas/vértices coincidentes de superfícies distintas são fundidos (STEP) e nós duplicados são unidos na malha, para que superfícies ligadas por uma aresta continuem conectadas (compartilhem nós) sob carga
- Fator de escala dos comprimentos (barra lateral): multiplica as medidas numéricas (comprimentos, análise) sem alterar a forma da geometria; útil para converter unidades (ex.: 1000 de m para mm)
- Triedro de referência (X, Y, Z) no canto do modelo 3D, que acompanha a rotação da câmera
- Ao selecionar uma aresta, o comprimento é exibido em mm (exato para STEP; soma dos segmentos da malha para STL)
- Deformada com fator de escala ajustável por barra de rolagem (1 = real; valor inicial deixa o deslocamento máx. em 5% da diagonal) e malha indeformada cinza de referência opcional
- Material (E, ν) e espessura uniforme são informados na barra lateral; use unidades consistentes (ex.: mm, N, MPa). Pressão positiva atua contra a normal externa (externa); negativa = pressão interna
- Tabela "Condições atribuídas" editável: valores de pressão, força nodal (FX/FY/FZ), corrosão (raio e espessura) e restrições das arestas (UX…RZ; célula vazia = GDL livre) podem ser alterados direto na tabela, e cada alteração já dispara um novo cálculo
- Força em arestas (total): ao selecionar arestas, informe a força total FX, FY, FZ (N, eixos globais); ela é distribuída uniformemente ao longo da aresta (com várias arestas, repartida pelo comprimento), entra no equilíbrio e no cálculo, e o valor de cada aresta é editável na tabela de condições
- Força nodal: no modo "Nó (força)" selecione um nó e informe FX, FY, FZ (em N, eixos globais); a carga entra no equilíbrio e no cálculo. Uma nova força no mesmo nó substitui a anterior; as forças são removidas ao mudar a malha ou em "Recomeçar"
- Corrosão: no modo "Nó (corrosão)" selecione um nó, informe o raio (qualquer valor positivo, em mm) e a espessura corroída (< nominal). Os elementos cuja distância sobre a superfície (geodésica pela malha) ao nó seja ≤ raio recebem a espessura reduzida; zonas sobrepostas usam a menor espessura. A zona aparece em magenta. Os defeitos são removidos ao mudar a malha (arquivo, densidade ou refino). A malha precisa ser fina o bastante para que a zona contenha elementos
- Nos resultados, cada área corroída recebe uma bandeirinha roxa com a razão (máx. Von Mises no meio da espessura dentro da área) / 220 MPa, em porcentagem (ex.: 50%), posicionada no ponto onde esse máximo ocorre; ao lado do valor aparece "Aprovado - ASME API-579 part 5 Nível 3" (≤ 100%) ou "Reprovado - ASME API-579 part 5 Nível 3" (> 100%)
- Botão "Recomeçar" (barra lateral): remove todas as cargas, condições de contorno, defeitos de corrosão, seleção e resultados, mantendo a geometria e a malha
- Isométrico em PDF: além de STEP/STL, aceita o PDF vetorial de um isométrico de tubulação (`isometric.py`). A linha de centro é o traço mais grosso; os trechos devem estar sobre os eixos isométricos (30°/90°/150°). A rosa dos ventos do desenho define N/E, e os comprimentos vêm das caixas de coordenadas (N/E/EL) ou, na falta delas, das cotas. DN, diâmetro externo e espessura são lidos do desenho (editáveis), assim como a pressão interna (tabela PRESSÃO (bar g), convertida para MPa; escolha Projeto ou Operação — padrão Projeto — e edite se necessário), que é aplicada automaticamente, com sinal negativo (pressão interna), em todas as superfícies da tubulação; os valores por superfície continuam editáveis na tabela de condições, assim como os comprimentos, numa tabela. As curvas 90° têm raio 1,5 × DN (ex.: 6" → 9" = 228,6 mm; o fator é editável) e a casca é gerada na superfície média (Ø = DE − e), com a espessura de parede já sugerida na análise. Limitações: linha sem ramais, uma única folha, PDF vetorial (não imagem)
- Gravidade: opção "Considerar gravidade (direção −Z)" na barra lateral (ativada por padrão) aplica o peso próprio da casca (densidade do aço × espessura local, já com a espessura corroída) como força nodal em −Z; "Tubulação cheia de água" (também ativada por padrão) soma o peso da água a partir do diâmetro interno (volume por área de parede = ri²/(2·rm), exato para cilindros e curvas). Padrões em mm, N, MPa, t: g = 9810 mm/s², aço 7,85e-9 t/mm³, água 1,0e-9 t/mm³. O equilíbrio estático considera essas cargas
- Validado contra soluções analíticas (viga em balanço sob pressão; cilindro fechado sob pressão externa)

## Executar

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
