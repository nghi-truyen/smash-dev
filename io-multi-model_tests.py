import smash
import numpy as np


setup, mesh = smash.load_dataset("cance")
model = smash.Model(setup, mesh)
model.run(inplace=True)

#generate the structure of the object: it is a dict of key:data to save: typeofstructure={light,medium,full}
keys_data=smash.generate_smash_object_structure(model,typeofstructure="medium")
print(keys_data)
#add a new data to save:
keys_data["parameters"].append('ci')

#Save a single smash model
smash.save_smash_model_to_hdf5("./model_light.hdf5", model, content="light", replace=True)
smash.save_smash_model_to_hdf5("./model_medium.hdf5", model, content="medium", replace=True)
smash.save_smash_model_to_hdf5("./model_full.hdf5", model, content="full", replace=True)
smash.save_smash_model_to_hdf5("./model_user.hdf5", model, keys_data=keys_data, replace=True)

#view the hdf5 file
hdf5=smash.io.multi_model_io.open_hdf5("./model_user.hdf5")
hdf5.keys()
hdf5["mesh"].keys()
hdf5["parameters"].keys()
hdf5["output"].keys()
hdf5["output"].attrs.keys()
hdf5["output/fstates"].keys()
hdf5["setup"].attrs.keys()
hdf5.close()


#save multi smash model at different place
smash.save_smash_model_to_hdf5("./multi_model.hdf5", model,location="model1",replace=True)
smash.save_smash_model_to_hdf5("./multi_model.hdf5", model,location="model2",replace=False)


hdf5=smash.io.multi_model_io.open_hdf5("./multi_model.hdf5")
hdf5.keys()
hdf5["model2"]["setup"].attrs.keys()
hdf5["model2"]["mesh"].keys()
hdf5["model2"]["output"].keys()
hdf5["model2"]["output"].attrs.keys()
hdf5.close()

#manually group different object in an hdf5
hdf5=smash.io.multi_model_io.open_hdf5("./model_subgroup.hdf5", replace=True)
hdf5=smash.io.multi_model_io.add_hdf5_sub_group(hdf5, subgroup="model1")
keys_data=smash.io.multi_model_io.generate_smash_object_structure(model,typeofstructure="medium")
smash.io.multi_model_io.dump_object_to_hdf5_from_iteratable(hdf5["model1"], model, keys_data)

hdf5=smash.io.multi_model_io.open_hdf5("./model_subgroup.hdf5", replace=False)
hdf5=smash.io.multi_model_io.add_hdf5_sub_group(hdf5, subgroup="model2")
keys_data=smash.io.multi_model_io.generate_smash_object_structure(model,typeofstructure="medium")
smash.io.multi_model_io.dump_object_to_hdf5_from_iteratable(hdf5["model2"], model, keys_data)

hdf5.keys()
hdf5["model1"].keys()
hdf5["model2"].keys()
hdf5.close()


#dump model object to a dictionnay
dictionary=smash.io.multi_model_io.dump_object_to_dictionary(model)


#load an hdf5 file to a dictionary
dictionary=smash.load_hdf5_file("./multi_model.hdf5")
dictionary["model1"].keys()
dictionary["model1"]["mesh"].keys()

#read only a part of an hdf5 file
hdf5=smash.io.multi_model_io.open_hdf5("./multi_model.hdf5")
dictionary=smash.io.multi_model_io.read_hdf5_to_dict(hdf5["model1"])
dictionary.keys()

#reload a full model object
model_reloaded=smash.load_hdf5_file("./model_full.hdf5",as_model=True)
model_reloaded
model_reloaded.run()


