import modal

LOCAL = False


def generate_passenger(survived, passenger_id):
    """
    Returns a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import numpy as np
    import random

   
    random_pclass = random.randint(1, 3)
    random_sex = random.randint(0, 1)


    bins = [-np.infty, 20, 25, 29, 30, 40, np.infty] # use same bins as in feature definition!
    age_bin_min = 0
    age_bin_max = len(bins) - 1

    random_age_bin = random.uniform(age_bin_min, age_bin_max)
   
    random_embarked = random.uniform(0, 2)
    
    random_parch= random.randint(0,6)
    random_sibsp= random.randint(0,8)





    passenger_df = pd.DataFrame({"passengerid": [passenger_id],"pclass": [random_pclass],"sex": [random_sex],"age": [random_age_bin],"sibsp": [random_sibsp],"parch":[random_parch],"embarked":[random_embarked]
        
    
                                 
                                 
                                 
                                 })

    passenger_df['survived'] = survived
    return passenger_df


def get_random_titanic_passenger(passenger_id):
    """
    Returns a DataFrame containing one random passenger
    """
    import random

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = 1
    if pick_random == 1:
        passenger_df = generate_passenger(1, passenger_id)
        print("Survived")
    else:
        passenger_df = generate_passenger(0, passenger_id)
        print("Didn't survive")

    return passenger_df


def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)

    # get max id
    titanic_df = titanic_fg.read()
    passenger_id = titanic_df['passengerid'].max() + 1

    passenger_df = get_random_titanic_passenger(passenger_id)
    titanic_fg.insert(passenger_df, write_options={"wait_for_job": False})


if not LOCAL:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4"])

    @stub.function(image=image, secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


if __name__ == "__main__":
    if LOCAL:
        g()
    else:
        with stub.run():
            f()
            