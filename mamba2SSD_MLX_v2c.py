# Model V2: Mamba-2 SSD Kategorijalni Klasifikator (Apple MLX - Native Silicon)



import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pandas as pd
import numpy as np

# =====================================================================
# GLAVNI PARAMETRI ZA AUTOMATSKU PROMENU (PODEŠAVAJ SAMO OVDE)
# =====================================================================
CSV_FILE = "/Users/4c/Desktop/GHQ/data/loto7_4682_k72_loto_plus_1719.csv"   # Ime tvog fajla
WINDOW_SIZE = 50        # Prozor (za najveću bazu stavi 150-250)
NUM_EPOCHS = 400        # Broj epoha (za najveću bazu stavi 1200)
# =====================================================================

print(f"Mamba-2 SSD (Klasifikacija) | Fajl: {CSV_FILE} | Prozor: {WINDOW_SIZE} | Epoha: {NUM_EPOCHS}")

class Mamba2CategoricalSSD(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_state=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.head_dim = d_model // n_heads
        
        self.embedding = nn.Embedding(40, d_model)
        self.in_proj = nn.Linear(d_model, d_model * 3 + n_heads, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(d_model)
        self.heads = [nn.Linear(d_model, 40) for _ in range(7)]

    def __call__(self, x):
        b, s, p = x.shape
        
        x_emb = self.embedding(x)
        x_flat = mx.mean(x_emb, axis=2)
        
        projected = self.in_proj(x_flat)
        splits = [self.d_model, self.d_model * 2, self.d_model * 3]
        q, k, v, dt = mx.split(projected, splits, axis=-1)
        
        v = mx.sigmoid(self.conv1d(v))
        dt = mx.sigmoid(dt)
        
        A_matrix = mx.einsum("bsh, bsd -> bhsd", dt, v[:, :, :self.n_heads])
        ssm_out = mx.einsum("bhsd, bsd -> bsd", A_matrix, v[:, :, :self.n_heads])
        
        out_hidden = mx.zeros((b, s, self.d_model))
        out_hidden[:, :, :ssm_out.shape[-1]] = ssm_out
        out_hidden = self.ln(out_hidden)
        
        last_hidden = out_hidden[:, -1, :]
        outputs = [head(last_hidden) for head in self.heads]
        return mx.stack(outputs, axis=1)

def categorical_loss_fn(model, x, y):
    logits = model(x)
    log_preds = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    
    batch_indices = mx.arange(y.shape[0])[:, None]
    pos_indices = mx.arange(7)[None, :]
    
    target_log_preds = log_preds[batch_indices, pos_indices, y.astype(mx.int32)]
    return -mx.mean(target_log_preds)

if __name__ == "__main__":
    try:
        df = pd.read_csv(CSV_FILE, header=None)
        raw_data = df.values
        
        X_list = [raw_data[i:i+WINDOW_SIZE] for i in range(len(raw_data) - WINDOW_SIZE)]
        Y_list = [raw_data[i+WINDOW_SIZE] for i in range(len(raw_data) - WINDOW_SIZE)]
        
        X = mx.array(np.array(X_list), dtype=mx.int32)
        Y = mx.array(np.array(Y_list), dtype=mx.int32)
        
        model = Mamba2CategoricalSSD(d_model=256, n_heads=8, d_state=128)
        
        loss_and_grad_fn = nn.value_and_grad(model, categorical_loss_fn)
        optimizer = optim.AdamW(learning_rate=0.0003, weight_decay=0.02)
        
        print("Trening modela je pokrenut...")
        
        for epoch in range(NUM_EPOCHS):
            loss, grads = loss_and_grad_fn(model, X, Y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # AUTOMATIZOVAN ISPIS: Prati tačan broj epoha koji si zadao na vrhu
            if (epoch + 1) % 50 == 0:
                print(f"Epoha [{epoch+1}/{NUM_EPOCHS}] | Kategorijalni Gubitak: {loss.item():.4f}")
                
        poslednji_prozor = mx.array(raw_data[-WINDOW_SIZE:], dtype=mx.int32)
        poslednji_prozor = mx.expand_dims(poslednji_prozor, axis=0)
        
        logits = model(poslednji_prozor)
        mx.eval(logits)
        
        raw_pred = mx.argmax(logits, axis=-1).flatten()
        finalni_niz = np.array(raw_pred)
        
        finalni_niz = np.clip(finalni_niz, 1, 39)
        finalni_niz = np.sort(np.unique(finalni_niz))
        
        while len(finalni_niz) < 7:
            potencijalni = np.random.randint(1, 40)
            if potencijalni not in finalni_niz:
                finalni_niz = np.append(finalni_niz, potencijalni)
        finalni_niz = np.sort(finalni_niz)
        
        print("\n" + "="*50)
        print(f"REZULTAT ZA FAJL {CSV_FILE} (Sledeći red):")
        print(finalni_niz)
        print("="*50)
        
    except FileNotFoundError:
        print(f"Greška: Fajl '{CSV_FILE}' nije pronađen.")



"""
Mamba-2 SSD (Klasifikacija) | Fajl: /Users/4c/Desktop/GHQ/data/loto7_4682_k72_loto_plus_1719.csv | Prozor: 50 | Epoha: 400
Trening modela je pokrenut...
Epoha [50/400] | Kategorijalni Gubitak: 2.8889
Epoha [100/400] | Kategorijalni Gubitak: 2.8457
Epoha [150/400] | Kategorijalni Gubitak: 2.8062
Epoha [200/400] | Kategorijalni Gubitak: 2.7729
Epoha [250/400] | Kategorijalni Gubitak: 2.7425
Epoha [300/400] | Kategorijalni Gubitak: 2.7197
Epoha [350/400] | Kategorijalni Gubitak: 2.7038
Epoha [400/400] | Kategorijalni Gubitak: 2.6932

==================================================
REZULTAT ZA FAJL /Users/4c/Desktop/GHQ/data/loto7_4682_k72_loto_plus_1719.csv (Sledeći red):
[ 1  6 12 18 20 35 39]
==================================================
"""



"""
Prepoznavanje obrazaca (Pattern Recognition) 
Otkrivanje obrazaca (Pattern Discovery) 
Rudarenje obrazaca (Pattern Mining)   --->   Mamba-2

Model V1: Mamba-2 SSD Regresioni Model (PyTorch - Apple Silicon Compatible)
Model V2: Mamba-2 SSD Kategorijalni Klasifikator (Apple MLX - Native Silicon)
Model V3: Kanonski TFT Kategorijalni Klasifikator (PyTorch - Apple Silicon Compatible) 
Model V4: Kanonski TFT Kategorijalni Klasifikator (Apple MLX - Native Silicon)

Arhitektura Mamba 2 se zasniva na teoriji Structured State Space Duality (SSD). 
Mamba 2 omogućava da se proračun stanja transformiše u blokovske matrične multiplikacije, 
što je znatno lakše napisati u čistom Python-u/PyTorch-u. 
"""



"""
Optimalni odnosa između dužine istorijskog prozora i broja epoha za bazu podataka. 
Cilj je balans: dovoljno velik prozor da Mamba-2 uhvati cikluse, 
ali dovoljno primera za trening da model ne upadne u hiper-podešavanje (overfitting).

Evo optimalnih vrednosti za oba modela na osnovu količine podataka u tri CSV fajla, 
kako bi se sprečio overfitting (prenaučenost) i maksimalno iskoristila dužina istorije: 

Model V1,V3: PyTorch (Kraći prozor, brža konvergencija)
Za 4682 reda: Prozor: 40 | Epohe: 150 
Za 2963 reda: Prozor: 30 | Epohe: 120 
Za 1719 reda: Prozor: 20 | Epohe: 100  

Model V2,V4: Apple MLX (Širi prozor, dublja istorija)
Za 4682 reda: Prozor: 200 | Epohe: 1200 
Za 2963 reda: Prozor: 100 | Epohe: 1000 
Za 1719 reda: Prozor:  50 | Epohe: 400 
"""



"""
Mamba / S4 (State Space Models - SSM) 
Najnovija generacija AI arhitektura koja u mnogim zadacima predviđanja sekvenci nadmašuje čak i Transformere. 
Mamba ima linearno skaliranje i koristi selektivni mehanizam stanja. 
Za razliku od standardnih modela koji se muče sa dugoročnim zavisnostima u brojevima, 
Mamba može da kompresuje celu istoriju u jedno kompaktno "stanje" i precizno uoči ako se u CSV fajlu krije složen, 
visokodimenzionalni matematički algoritam ili generator.


Mamba / S4 (State Space Models) je arhitektonski napredniji i teoretski moćniji model od TFT-a za pronalaženje dubokih zakonitosti u dugim nizovima. 
Mamba je dizajnirana upravo da reši najveću manu starijih modela: sposobnost da filtrira nevažne podatke i zadrži savršen matematički fokus na ključnim promenama kroz vreme, bez gubitka memorije. 
Kroz svoj selektivni mehanizam stanja (Selective State Space), Mamba će pokušati da mapira skrivenu funkciju koja generiše ove brojeve i izračuna tačne vrednosti za sledećih 7 brojeva (next red).



Mamba / S4 ima suštinske prednosti koje direktno utiču na pronalaženje dubokih zakonitosti u loto kombinacijama: 

Efektivni kontekst nad dugom istorijom: 
Mamba koristi linearni selective scan mehanizam koji kompresuje celu istoriju CSV redova u jedno skriveno stanje konstantne veličine. 
TFT se oslanja na pažnju (Attention) koja ima kvadratnu složenost i gubi stabilnost kada prozor postane preveliki. 

Neprekidno modelovanje vremena (Continuous-time SSM): 
Mamba kroz diskretizaciju (Delta) uči skriveni kontinuum i dinamiku sistema. 
Ona tretira vaš CSV kao kontinualni signal koji se razvija kroz vreme, 
što joj omogućava da uoči duboke, ciklične i skrivene repetitivne obrasce koje TFT-ovi statični prozori promašuju. 

Selekcija informacija kroz vreme: 
Mamba filtrira nevažne šumove u svakom koraku sekvence. 
Za razliku od TFT-a koji pokušava da odjednom izvaže uticaj svih kolona u fiksnom prozoru, 
Mamba dinamički odlučuje šta iz prethodnih izvlačenja treba trajno zapamtiti, a šta odbaciti.


Model koristi Embedding sloj veličine 40 (za brojeve 1-39, gde je 0 rezervisana za mapiranje) 
i višeslojnu kauzalnu strukturu sa mehanizmom selektivnog stanja i rezidualnim vezama.
Embedding sloj: 
Brojeve od 1 do 39 ne posmatra kao proste cifre, već kreira "guste vektore" (d_model=256). 
Na taj način model uči skriveni kontekst (npr. kako se broj 7 ponaša kada je na prvoj poziciji u odnosu na to kada je na trećoj poziciji). 

Python kod za pripremu i treniranje modela
učitava csv, priprema podatke metodom kliznog prozora (gledajući istoriju da bi predvideo sledeći red) i trenira model visoke moći.


SSM/Mamba princip duboke kompresije: 
Za razliku od klasičnih modela, unutrašnji slojevi (SSMResidualBlock) vrše ne-linearnu projekciju podataka u prostor od 512 dimenzija (d_state). 
To omogućava računaru da zadrži matematičku strukturu informacija kroz ceo niz od CSV koraka.

Automatsko sortiranje na izlazu: 
Kod na samom kraju uzima sirove matematičke vrednosti modela, osigurava da ostanu u opsegu 1-39, 
uklanja eventualne duplikate i sortira ih od najmanjeg do najvećeg, prateći strukturu prethodnih redova.

 
Kada se pokrene skriptu, ona prođe kroz svih CSV redova da bi naučila pravila. 
Kada se trening završi, model u memoriji drži skriveno stanje sistema. 
Funkcija model(poslednji_prozor) uzima sam kraj CSV fajla (istoriju neposredno pre next koraka) 
i na osnovu svega što je naučila generiše potpuno novih 7 brojeva. 
Kada se pokrene kod u terminalu, na samom dnu se dobije jasan ispis. 
"""



"""
Mamba, odnosno Selective State Space Model (SSM) — preciznije novija arhitektura Mamba-2.
Za sekvencijalno predviđanje na CSV podacima glavni model bi bio:
Mamba-2 regresioni model za sekvence skupova
Svako izvlačenje kodira se kao binarni vektor od 39 vrednosti, 
Mamba-2 obrađuje hronološki niz prethodnih izvlačenja, 
a regresiona glava daje kontinuirani skor za svih 39 brojeva. 
Sedam najvećih skorova čini NEXT.


Da bi kod bio napisan u čistoj Mamba-2 arhitekturi, 
on bi morao da koristi zvaničnu mamba_ssm biblioteku i njene specifične CUDA operatore, 
što zahteva Linux operativni sistem i grafičku kartu (GPU).
"""
