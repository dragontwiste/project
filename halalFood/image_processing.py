def food_classification(image):
    
    import cv2 
    import pytesseract as pyt
    import numpy as np
    import matplotlib.pyplot as plt
    from thefuzz import process
    import pandas as pd
    from keras.utils import img_to_array

    img_float32 = np.float32(image)
    image= cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)

    #resizing the image
    new_width=800
    aspect_ratio=image.shape[1]/image.shape[0]
    new_height=int(new_width/aspect_ratio)
    image=cv2.resize(image,(new_width,new_height))

    #image binarzation (black and white image)
    image=img_to_array(image, dtype='uint8')
    image=cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    #creating sharpening kernel
    sharpening_kernel=np.array([[-1,-1,-1],
                            [-1,9,-1],
                            [-1,-1,-1]],np.float32)

    #sharpening the image with the filter
    image=cv2.filter2D(np.float32(image),-1,sharpening_kernel)

    #adjusting brightness and contrast
    image=cv2.convertScaleAbs(np.float32(image),alpha=1.4,beta=20)

    #reducing noise
    image=cv2.fastNlMeansDenoising(np.uint8(image),None,h=5,templateWindowSize=7,searchWindowSize=21)

    #dilation followed by erosion
    kernel=np.ones((2,2),np.uint8)
    result=cv2.erode(image,kernel,iterations=1)
    result=cv2.dilate(result,kernel,iterations=1)
    image=cv2.erode(result,kernel,iterations=1)

    #image deskewing(rotation)
    edges = cv2.Canny(image, 50, 150)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize the rotation angle
    angle = 0.0
    
    # Loop over the contours
    for contour in contours:
        # Fit a rotated bounding box around the contour
        box = cv2.minAreaRect(contour)
        angle = box[-1]
    
    # Deskew the image by rotating it by the calculated angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows), flags=cv2.INTER_LINEAR)

    #converting the image to text
    ocr_result= pyt.image_to_string(image)

    #defining halel and haram ingredients
    halel_ingredients="Agar, Algin, Allspice, Amylase, Anise, Annatto, Arrowroot, Baker's Yeast, Baker's Yeast Extract, Basil, Bay Leaves, Beta-Carotene, Bran, Bulgur wheat, Cocoa Butter, Caffeine, Canola Oil, Caramel, Caraway, Cardamom, Carnauba Wax, Carob, Carob Gum, Carrageenan, Cheese & Dairy Culture, Chives, Chocolate, Chocolate Liquor, Cinnamon, Citrus Oil, Clove, Cocoa Powder, Coconut, Coriander, Corn, Corn Bran, Corn Flour, Corn Gluten, Corn Meal, Corn Oil, Corn Starch, Corn Syrup, Corn Syrup Solids, Cranberry, Cumin, Curry Powder, Dextrin, Dextrose, Dill, Durum Flour, Enzyme, Farina, Fennel, Fenugreek, Flour, Fructose, Garlic, Gellan Gum, Ghatti Gum, Ginger, Gluten, Glycerin, Graham Flour, Grape, Guar Gum, High Fructose Corn Syrup, Honey, Horseradish, Hydrogenated Vegetable Oil, Hydrolyzed Vegetable Protein, Invert Sugar, Karaya, Kelp, Licorice Root, Locust Bean Gum, Mace, Maize, Malt, Maltodextrin, Maltose, Maple Sugar, Maple Syrup, Marjoram, Margarine, Modified Food Starch, Molasses, Mustard, Nutmeg, Oat, Oat Flour, Oatmeal, Olive Oil, Onion, Oregano, Palm Oil, Papa in, Paprika, Parsley, Peanut, Pectin, Peppers, Polydextrose, Potato Starch, Pregelatinized Starch, Protease Enzyme, Psyllium, Raisin, Rapeseed, Rennet, Rice, Rye, Safflower, Saffron, Sage, Savory, Semolina, Sesame, Shallot, Sodium Lauryl Sulfate, Sorbitol, Soybean, Soya Flour, Soya Oil, Soya SauceSpice, Starch, Sugar, Sunflower, Tapioca Starch, Textured Vegetable Protein, Tocopherol, Tofu, Tragacanth, Turola Yeast, Turmeric, Vanilla Bean, Vital Wheat, Wheat, Wheat Flour, Xanthan Gum, Xylitol, zein, Acetic acid, Alum, Aluminum Ammonium Sulfate, Ammonium, Ascorbic acid, Azodicarbonamide, Dry Artificial Colors, Artificial Flavors, Aspartame, Baking Powder, Baking Soda, Beeswax, Benzaldehyde, Benzoic acid, BHA, BHT, Calcium, Carboxymethylcellulose, Chlorine, Citric Acid, Cream of Tarter, L-Cysteine, Dicalcium Phosphate, Dipotassium Phosphate, EDTA, Dry FD&C Colors only, Ferric Oxide, Ferrous Sulphate, Folic Acid, Fumaric Acid, Glucon-Delta-Lactone, Glycerin, Iron, Lactic Acid, Magnesium Carbonate, Magnesium Stearate, Methyl Cellulose, Monocalcium Phosphate, Monosodium Glutamate, Fumaric Acid, Glucon-Delta-Lactone, Glycerin, Iron, Lactic Acid, Magnesium Carbonate, Magnesium Stearate, Methyl Cellulose, Monocalcium Phosphate, Monosodium Glutamate, Natural Flavoring, Natural Flavors, Niacin, Oleoresins of spices, Phosphates, Phosphoric Acid, Potassium, Propionic Acid, Propylene Glycol, Propyle Gallate, Riboflavin, Saccharine, Salt, Sea Salt, Silicon Dioxide, Sodium Acetate, Sodium Acid Pyrophosphate, Sodium Aluminum Phosphate, Sodium Aluminum Sulfate, Sodium Benzoate, Sodium Bicarbonate, Sodium Carbonate, Sodium Citrate, Sodium Hydroxide, Sodium Lactate, Sodium Lauryl Sulfate, Sodium Metabisulfate, Sodium Nitrate, Sodium Nitrite, Sodium sorbate, Sorbic Acid, Tartaric acid, TBHQ, Thiamine Mononitrate, Titanium Dioxide, Tricalcium phosphate, Vanillin, Vinegar, Acid Casein, Butter fat Lipolyzed, Buttermilk Solids, Caseinates , Rennet Casein, Cheese Powder, Cream, Cultured Milk, Cultured Cream Lipolyzed, Dried Milk, Lactose, Nonfat Dry milk, Sour Cream Solids, Reduced Mineral Whey, Whey, Whey Protein Concentrate, Calcium Stearate, Calcium Stearoyl Lactylate, DATEM, Diglyceride, Ethoxylated Mono- and Diglycerides, Glycerin, Glycerol Ester, Glycerol Monostearate, Hydroxylated Lecithin, Soya Lecithin, Enzyme Modified Soya Lecithin, Soybean Oil, Margarine, Mono- and Diglycerides, Monoglyceride, Partially Hydrogenated Vegetable Shortening, Polyglycerol Esters of Fatty Acids, Polyoxythylene Sorbitan Monostearate, Polysorbate 60, Polysorbate 65, Polysorbate 80, Propylene Glycol Monostearate, Sodium Stearoyl Lactylate, Softener, Sorbitan Monostearate, Tocopherol, Vegetable Oils, Albumin, Eggs, Egg White, Egg Yolks, Gelatin, Lecithin"
    halel_ingredients=halel_ingredients.split(", ")
    haram_ingredients="Alcohol, spices, Bear, Bear Flavor, Bear Batters, Fermented Cider, Hard Cider, Rum, Torula Yeast grown on liquor, Soya Sauce, Sherry Wine, Vanilla Extract, Wine, Wine vinegar, L-Cysteine, Bacon, Ham, Gelatin, Marshmallow containing pork gelatin, Pork, Bacon bits, Rennin, Pepsin, Beta-Carotene, pork enzymes, BHA & BHT, Butter fat Lipolyzed, Buttermilk Solids, Caseinates, Rennet Casein, Cheese Powder, Cultured Cream Lipolyzed, Cultured Milk, Lactose, Sour Cream Solids, Reduced Mineral Whey, Rennet, Whey, Whey Protein Concentrate, Calcium Stearate, Calcium Stearoyl Lactylate, DATEM, Diglyceride, Ethoxylated Mono- and Diglycerides, Glycerin, Glycerol Ester, Glycerol Monostearate, Hydroxylated Lecithin, Lard, Margarine, Mono- and Diglycerides, Monoglyceride, Polyglycerol Esters of Fatty Acids, Polyoxythylene Sorbitan Monostearate, Polysorbate 60, Polysorbate 65, Polysorbate 80, Propylene, Glycol Monostearate, Sodium Stearoyl Lactylate, Softener, Sorbitan Monostearate, Tocopherol"
    haram_ingredients=haram_ingredients.split(", ")
    ingredients=halel_ingredients+haram_ingredients

    #correcting misspelling
    ocr_result_split=ocr_result.split(" ,")
    choices=pd.Series(ingredients)
    for i in range(len(ocr_result_split)):
        L=process.extract(ocr_result_split[i],choices,limit=1)
        if L[0][1] >= 0.8 :
            ocr_result_split[i]=L[0][0]

    #veryfying haram ingredients
    halal=True
    for i in ocr_result_split:
        if i in haram_ingredients:
            halal=False
            break
    return(halal)
