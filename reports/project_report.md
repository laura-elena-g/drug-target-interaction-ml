## 1. Formularea problemei

Predicția interacțiunilor dintre medicamente și ținte biologice (DTI) este folosită în descoperirea computațională de medicamente pentru a restrânge mai eficient lista de compuși care merită testați experimental. În practică, într-un scenariu de virtual screening, nu contează doar dacă modelul etichetează corect o interacțiune ca activă sau inactivă. Contează și cum ordonează perechile compus–țintă, astfel încât cele cu șanse reale de activitate să ajungă cât mai sus în clasament.

În acest proiect, problema DTI a fost tratată ca o sarcină de clasificare binară, folosind setul de date KIBA. Fiecare exemplu descrie o pereche medicament–țintă prin structura compusului, secvența proteinei și un scor de afinitate. Pentru a obține etichete binare, perechile cu afinitate mai mare sau egală cu 12 au fost considerate active, iar cele sub acest prag au fost considerate inactive. Astfel, un semnal continuu a fost transformat într-o problemă de clasificare potrivită pentru modele clasice de machine learning.

Proiectul a urmărit două obiective. Primul a fost construirea unor baseline-uri solide, bazate pe feature-uri generate manual pentru compuși și proteine. Al doilea a fost compararea a două moduri de evaluare: split aleator și split la nivel de medicament. În al doilea caz, compușii din setul de antrenare nu mai apar deloc în setul de test. Diferența dintre cele două abordări contează, pentru că split-ul aleator poate da impresia unei performanțe mai bune decât cea pe care modelul ar avea-o în condiții mai apropiate de utilizarea reală.

Prin urmare, întrebarea proiectului nu a fost doar cât de bine clasifică modelul, ci și cât de relevantă rămâne această performanță atunci când trebuie să generalizeze către molecule nevăzute. Din punct de vedere aplicativ, asta este situația care interesează cel mai mult.

## 2. Descrierea setului de date

Pentru acest proiect a fost folosit setul de date KIBA, unul dintre benchmark-urile cunoscute pentru predicția interacțiunilor medicament–țintă. Varianta procesată utilizată aici conține 118.254 de perechi medicament–țintă. Fiecare rând include structura compusului, secvența proteinei și un scor asociat afinității. Coloanele păstrate pentru modelare au fost compound_iso_smiles, target_sequence și affinity.

Coloana compound_iso_smiles reprezintă structura moleculară a compusului sub formă textuală, iar target_sequence conține secvența de aminoacizi a proteinei. Coloana affinity descrie intensitatea interacțiunii și a fost folosită pentru derivarea etichetelor binare. Perechile cu valoare mai mare sau egală cu 12 au fost etichetate ca active, iar restul ca inactive.

După această transformare, distribuția claselor a rămas dezechilibrată: aproximativ 21,2% dintre exemple aparțin clasei active, iar 78,8% clasei inactive. Din acest motiv, metrici precum PR-AUC sau enrichment factor sunt mai utile decât simpla acuratețe.

Datele au fost evaluate în două moduri. În primul caz, exemplele au fost împărțite aleator în train și test. În al doilea caz, împărțirea s-a făcut la nivel de compus, astfel încât același medicament să nu apară în ambele subseturi. Acest al doilea scenariu este mai dificil, dar descrie mai bine situația în care modelul trebuie să facă predicții pentru molecule noi.

## 3. Ingineria caracteristicilor

Pentru compuși, feature-urile au fost generate din șirurile SMILES cu ajutorul RDKit. Fiecare moleculă a fost descrisă printr-un fingerprint Morgan cu rază 2 și 1024 biți. La acești biți au fost adăugați încă cinci descriptori moleculari: masa moleculară, LogP, numărul de donori de legături de hidrogen, numărul de acceptori de legături de hidrogen și aria polară topologică. În total, fiecare compus a fost reprezentat prin 1029 caracteristici.

Pentru proteine s-a folosit o reprezentare intenționat simplă. Fiecare secvență a fost codificată prin compoziția în aminoacizi, adică frecvența relativă a celor 20 de aminoacizi standard. În plus, a fost adăugată lungimea secvenței. Astfel, fiecare proteină a fost descrisă prin 21 de caracteristici.

Vectorul final pentru fiecare pereche medicament–țintă a rezultat prin concatenarea caracteristicilor de pe partea de compus cu cele de pe partea de proteină. Matricea finală de intrare a avut forma (118254, 1050).

Alegerea acestui tip de reprezentare a fost deliberată. Scopul a fost construirea unor baseline-uri clasice, clare și reproductibile, fără modele profunde sau embedding-uri preantrenate.

## 4. Abordarea de modelare

Au fost comparate două tipuri de modele: regresie logistică și XGBoost.

Regresia logistică a fost folosită ca baseline liniar. Pentru că feature-urile provin din surse diferite și au scale numerice diferite, înainte de antrenare s-a aplicat standardizarea datelor. Dezechilibrul claselor a fost tratat prin ponderi echilibrate.

XGBoost a fost ales ca baseline neliniar mai puternic. Modelul a fost antrenat cu 300 de estimatori, adâncime maximă 6, rată de învățare 0,1, subsample = 0.8 și colsample_bytree = 0.8. Ponderea clasei pozitive a fost ajustată în funcție de raportul dintre exemplele inactive și active din datele de antrenare.

Ambele modele au fost evaluate în două scenarii. În split-ul aleator, datele au fost împărțite stratificat 80/20. În split-ul la nivel de medicament, compușii unici au fost separați mai întâi între train și test, iar toate perechile asociate unui anumit compus au rămas într-o singură partiție.

Evaluarea s-a bazat în principal pe ROC-AUC și PR-AUC. În plus, au fost calculate matrici de confuzie la pragul 0,5. Pentru că proiectul are și o componentă de screening, au fost incluse și metrici bazate pe ordonarea predicțiilor, precum Precision@K, Recall@K și enrichment factor.


## 5. Resultate

Rezultatele au arătat diferențe clare atât între modele, cât și între strategiile de evaluare.

Pentru regresia logistică, split-ul aleator a dus la un ROC-AUC de 0,805 și un PR-AUC de 0,574. Când evaluarea s-a făcut la nivel de medicament, scorurile au scăzut la 0,659 pentru ROC-AUC și 0,358 pentru PR-AUC. Scăderea sugerează că modelul liniar se bazează destul de mult pe tipare care nu se transferă bine către compuși noi.

XGBoost a avut rezultate mai bune în ambele scenarii. În split-ul aleator, modelul a atins un ROC-AUC de 0,908 și un PR-AUC de 0,758. În split-ul la nivel de medicament, performanța a rămas bună: 0,865 pentru ROC-AUC și 0,667 pentru PR-AUC. Asta arată că modelul neliniar reușește să capteze relații pe care regresia logistică le surprinde mai slab.

În ambele cazuri, split-ul aleator a produs scoruri mai mari decât split-ul la nivel de medicament. Acest lucru sugerează că evaluarea aleatoare poate supraestima performanța reală a modelului. În schimb, split-ul la nivel de medicament oferă o imagine mai apropiată de comportamentul pe molecule nevăzute anterior.

Per ansamblu, XGBoost a fost modelul cu cele mai bune rezultate. Dintre toate variantele testate, cea mai relevantă pentru un scenariu realist rămâne versiunea evaluată pe split la nivel de medicament.


## 6. Interpretarea din perspectiva screening-ului

Pentru că proiectul a fost gândit și din perspectiva virtual screening-ului, analiza nu s-a oprit la metricile globale de clasificare. A fost urmărit și felul în care modelele ordonează perechile compus–țintă, astfel încât interacțiunile active să apară cât mai sus în listă.

Cea mai bună performanță în acest sens a fost obținută de XGBoost în scenariul cu split la nivel de medicament. Pentru acest model, Precision@5% a fost 0,866. Cu alte cuvinte, aproape 86,6% dintre predicțiile aflate în primele 5% din clasament au fost active. Enrichment factor la 5% a fost 4,12, ceea ce arată o îmbogățire clară a părții superioare a clasamentului în exemple active.

Tendința s-a păstrat și la praguri mai mari. Pentru XGBoost în split-ul la nivel de medicament, Precision@10% a fost 0,771, iar Precision@20% a fost 0,637. Valorile acestea arată că modelul reușește să aducă un număr mare de interacțiuni active în partea de sus a listei chiar și atunci când este testat pe compuși noi.

Regresia logistică a avut un comportament mai slab, mai ales în scenariul cu split la nivel de medicament, unde Precision@5% a fost 0,474, iar EF@5% a fost 2,25. Prin urmare, XGBoost a fost mai potrivit și pentru prioritizarea candidaților, nu doar pentru clasificare generală.

## 7. Importanța caracteristicilor

Analiza importanței caracteristicilor a fost realizată pentru modelul XGBoost antrenat și evaluat în scenariul cu split la nivel de medicament, deoarece acesta a oferit cea mai relevantă combinație între performanță și realismul evaluării.

Distribuția importanțelor a arătat o dependență foarte puternică de feature-urile asociate compușilor. Aproximativ 98,15% din importanța totală a fost atribuită caracteristicilor de pe partea de medicament, în timp ce doar 1,85% a revenit caracteristicilor de pe partea de proteină. Rezultatul sugerează că modelul extrage aproape tot semnalul predictiv din reprezentarea ligandului.

Primele 20 de caracteristici ca importanță au provenit exclusiv din blocul de feature-uri al compușilor. Cele mai multe au fost dimensiuni din fingerprint-ul Morgan, ceea ce indică faptul că modelul s-a bazat în principal pe informație structurală de tip substructură. Printre caracteristicile din top a apărut și descriptorul pentru numărul de donori de legături de hidrogen, semn că modelul a folosit și o parte din informația fizico-chimică explicită.

Rezultatul arată că reprezentarea compușilor este suficient de puternică pentru a susține predicții utile într-un cadru clasic de machine learning. În același timp, sugerează și o limitare clară a pipeline-ului: reprezentarea proteinelor este prea simplă pentru a contribui substanțial. O compoziție globală în aminoacizi și lungimea secvenței oferă doar o descriere foarte generală și nu surprind motive locale, domenii sau context structural.


## 8. Limitări și direcții viitoare

Proiectul are câteva limite clare. Prima ține de reprezentarea proteinelor. Folosirea compoziției în aminoacizi și a lungimii secvenței oferă o descriere destul de grosieră și lasă în afara modelului informații legate de ordine locală, motive funcționale, organizare structurală sau regiuni de legare.

A doua limită este dată de tipul de reprezentări folosite. Modelele au fost antrenate pe descriptori definiți manual, nu pe reprezentări învățate. Alegerea a fost intenționată, pentru că proiectul și-a propus baseline-uri clasice și ușor de interpretat. Totuși, această alegere reduce cantitatea de informație structurală și secvențială pe care modelul o poate exploata.

O altă limită ține de evaluare. Analiza s-a concentrat pe generalizarea către compuși nevăzuți, dar nu a inclus scenarii și mai dificile, cum ar fi generalizarea către ținte nevăzute sau split-uri complet cold-start, unde nici compusul, nici ținta nu apar în antrenare.

Chiar și cu aceste limite, proiectul evidențiază clar un lucru: modul în care este făcută evaluarea schimbă substanțial imaginea asupra performanței. Split-urile aleatoare tind să dea rezultate mai bune pe hârtie, în timp ce split-urile la nivel de medicament oferă o estimare mai apropiată de situațiile practice. În continuare, o direcție firească ar fi îmbunătățirea reprezentării proteinelor și extinderea evaluării către scenarii de generalizare mai stricte.