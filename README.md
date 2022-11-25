Experiments with calibration and concept drift

model brez calibration
model s calibration na calibration set
model, ki se na novo trenira ko zaznamo drift
model, ki se na novo calibrira, ko zaznamo drift
model, ki se na novo trenira in kalibrira, ko zaznamo drift
model, ki se stalno kalibrira z novimi napovedmi


ta pool set je test set v resnici: ga krčimo po eno točko na enkrat, na preostalem pa meriš performance (AUC)
bi pa rekel da narediva to dokler ostane x primerov v train set (eno smiselno število, kjer je še zadosti samples za vsak class, npr.: zadnjih 10%, ali kaj takega)


sicer pa lahko tudi imamo eno strategijo, kjer dodajaš še nekaj starejših podatkov in imaš moving window, dokler tistih 10% pade od warning level naprej
ko imaš že samo podatke od warning naprej, je to to