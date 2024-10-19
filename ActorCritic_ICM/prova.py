class aereo():
    def __init__(self, posti, passeggeri):
        self.posti_totali = posti
        self.business = posti / 3
        self.economy = self.posti_totali - self.business
        self.lista_posti = [i for i in range(self.posti_totali)]
        self.passeggeri = [i for i in range(passeggeri)]
        self.info = []
        
    def assegna(self, posto, passeggero):
        self.info.append([posto, passeggero])
        
    def lista_passeggeri(self):
        print(f"lista passeggeri: {self.info}")
        
    def cancella(self, passeggero):
        el = [pos for pos, pas in enumerate(self.info) if pas[1] == passeggero]
        print(el)
        del self.info[el[0]]
        
    
    
a = aereo(10, 5)
print(a.passeggeri, a.lista_posti)


a.assegna(3, 4)
a.lista_passeggeri()
a.assegna(1, 3)
a.lista_passeggeri()
a.cancella(4)
a.lista_passeggeri()
        
        
def stampa(x):
    print("ciao")

stampa(10)        