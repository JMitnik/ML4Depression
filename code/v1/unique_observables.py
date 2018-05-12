from exploration import module_data

# 'intro'/ 'edumotiv'/ 'ba'/ 'cr'/ 'ex'/ 'pst'/'eval'
unique_modules = module_data['moduleId'].unique()

# 'loginW', 'entryPage', 'exitPage', 'loginM', 'logoutM', 'logoutW'
# Presumable, where W is web, and M is mobile
unique_operation_types = module_data['operationType'].unique()
