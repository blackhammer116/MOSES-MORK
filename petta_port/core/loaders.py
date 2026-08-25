import csv
import os

def load_data_and_inject(filepath: str, metta_space_ref, output_col='O'):
    """
    Reads a CSV and injects it into the MeTTa space as (TruthValue Sym (:: True False ...)) 
    using MeTTa's Python interoperability.
    """
    data_columns = {}
    target = []
    
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if not headers: return False
            
            for h in headers:
                if h != output_col:
                    data_columns[h] = []
                    
            for row in reader:
                for key, val in row.items():
                    v = val.strip().upper()
                    bool_val = v in ('1', 'TRUE', 'T', 'YES')
                    if key == output_col:
                        target.append(bool_val)
                    else:
                        data_columns[key].append(bool_val)
                        
        # Construct the MeTTa string additions
        def to_metta_list(py_list):
            if not py_list:
                return "()"
            res = "()"
            for b in reversed(py_list):
                res = f"(:: {'True' if b else 'False'} {res})"
            return res

        for key, cols in data_columns.items():
            metta_str = f"(TruthValue {key} {to_metta_list(cols)})"
            metta_space_ref.add_atom(metta_str)
            print(f"Loaded Knob: {key}")

        metta_target = f"(TruthValue Target {to_metta_list(target)})"
        metta_space_ref.add_atom(metta_target)
        print("Loaded Target.")
        return True
        
    except Exception as e:
        print(f"Error loading CSV for MeTTa: {e}")
        return False
