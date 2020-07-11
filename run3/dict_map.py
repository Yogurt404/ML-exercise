class ClassMap:
    def __init__(self):
        self.l = ['bedroom', 
                  'Coast', 
                  'Forest', 
                  'Highway', 
                  'industrial', 
                  'Insidecity', 
                  'kitchen', 
                  'livingroom', 
                  'Mountain', 
                  'Office', 
                  'OpenCountry', 
                  'store', 
                  'Street', 
                  'Suburb', 
                  'TallBuilding']
    
    def get_str(self, i):
        return self.l[i]

    def get_idx(self, s):
        return self.l.index(s)