from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16  # Import pretrained ViT
from PIL import Image
import io



# Load the trained model
MODEL_PATH = "best_pt_vit_model2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES=8

# Load Pretrained Vision Transformer (ViT)
model = vit_b_16(weights="IMAGENET1K_V1")  # Load ViT with pretrained ImageNet weights
model.heads = nn.Linear(model.hidden_dim, NUM_CLASSES)  # Replace classification head

model.to(DEVICE)

print(f"Loading model on device: {DEVICE}")
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Changed from 128 to 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


cocoa_blackpod={
  "disease": "Black Pod Disease",
  "causes": [
    "Fungal infection by Phytophthora species (P. palmivora, P. megakarya, P. citrophthora)",
    "High humidity and excessive rainfall",
    "Poor air circulation due to dense planting",
    "Infected pods and trees serving as a source of spores",
    "Soil and insect transmission",
    "Weak plant immunity due to stress or malnutrition"
  ],
  "remedies": {
    "organic": {
      "pruning_and_spacing": "Improve air circulation to reduce moisture buildup",
      "regular_sanitation": "Remove and destroy infected pods away from the plantation",
      "mulching_and_drainage": "Reduce excess moisture to prevent fungal growth",
      "biological_control": {
        "Trichoderma": "Apply Trichoderma spp. to suppress Phytophthora",
        "compost_tea": "Use compost tea to boost plant immunity"
      },
      "neem_extracts": "Spray neem oil or neem leaf extracts to slow fungal spread",
      "copper_based_fungicides": "Use Bordeaux mixture (copper sulfate + lime) for organic control"
    },
    "inorganic": {
      "copper_based_fungicides": [
        "Bordeaux mixture",
        "Copper oxychloride",
        "Copper sulfate"
      ],
      "phosphonate_based_fungicides": [
        "Potassium phosphonate to boost plant resistance"
      ],
      "systemic_fungicides": [
        "Metalaxyl",
        "Ridomil Gold"
      ],
      "preventive_spraying": "Apply fungicides before peak rainy seasons"
    }
  },
  "best_practices": [
    "Combine organic and inorganic control methods",
    "Monitor fields regularly for early symptoms",
    "Use resistant cocoa varieties if available"
  ]
}

cocoa_frostypod={
  "disease": "Frosty Pod Disease",
  "causes": [
    "Fungal infection by Moniliophthora roreri",
    "High humidity and excessive rainfall",
    "Spores spread through wind, rain, and contaminated tools",
    "Poor farm sanitation with infected pods left in the field",
    "Lack of resistant cocoa varieties",
    "Weak plant immunity due to environmental stress"
  ],
  "remedies": {
    "organic": {
      "pruning_and_spacing": "Improve air circulation to reduce humidity levels",
      "regular_sanitation": "Remove and destroy infected pods far from the plantation",
      "shade_management": "Reduce excessive shading to lower humidity and fungal spread",
      "biological_control": {
        "Trichoderma": "Apply Trichoderma spp. to suppress Moniliophthora roreri",
        "Bacillus_based_control": "Use Bacillus subtilis to inhibit fungal growth"
      },
      "botanical_extracts": "Use neem oil or plant-based antifungals (garlic, chili extract)",
      "organic_fertilization": "Apply compost and organic fertilizers to strengthen plant immunity"
    },
    "inorganic": {
      "copper_based_fungicides": [
        "Bordeaux mixture (copper sulfate + lime)",
        "Copper oxychloride",
        "Copper hydroxide"
      ],
      "systemic_fungicides": [
        "Propiconazole",
        "Tebuconazole"
      ],
      "preventive_spraying": "Apply fungicides before peak rainy seasons to reduce infection risk"
    }
  },
  "best_practices": [
    "Regularly monitor for early symptoms of infection",
    "Use a combination of organic and chemical control methods",
    "Maintain proper pruning, spacing, and sanitation",
    "Consider planting resistant cocoa varieties if available"
  ]
}

cocoa_mirid={
  "disease": "Mirid Disease (Capsid Bugs Infestation)",
  "causes": [
    "Infestation by mirid insects (Sahlbergella singularis, Distantiella theobroma)",
    "Mirids suck sap from cocoa pods and stems, injecting toxic saliva",
    "Favorable hot and dry conditions promote mirid outbreaks",
    "Poor farm sanitation, allowing mirids to breed in overgrown cocoa farms",
    "Absence of natural predators due to overuse of broad-spectrum pesticides",
    "Lack of regular monitoring, leading to unnoticed infestations"
  ],
  "remedies": {
    "organic": {
      "pruning_and_sanitation": "Regularly prune trees and remove overgrown branches to reduce mirid hiding spots",
      "biological_control": {
        "predatory_insects": "Encourage natural predators like ants, spiders, and parasitic wasps",
        "entomopathogenic_fungi": "Apply Beauveria bassiana to infect and kill mirids"
      },
      "botanical_sprays": [
        "Neem oil to disrupt mirid reproduction",
        "Garlic or chili-based extracts as natural repellents"
      ],
      "handpicking_and_trapping": [
        "Physically remove mirids from cocoa trees",
        "Use yellow sticky traps to monitor and reduce mirid populations"
      ]
    },
    "inorganic": {
      "selective_insecticides": [
        "Lambda-cyhalothrin (Pyrethroid-based insecticide)",
        "Acetamiprid (Neonicotinoid-based insecticide)"
      ],
      "systemic_insecticides": [
        "Imidacloprid to prevent mirid infestations",
        "Thiamethoxam for long-term control"
      ],
      "preventive_spraying": "Apply insecticides at the beginning of dry seasons when mirids are most active"
    }
  },
  "best_practices": [
    "Regularly inspect cocoa farms for early signs of mirid damage",
    "Integrate biological and selective chemical control methods",
    "Encourage natural predators to maintain ecological balance",
    "Implement farm hygiene by removing infested pods and branches"
  ]
}

coffee_browneye={
  "disease": "Browneye Disease (Coffee Berry Disease - CBD)",
  "causes": [
    "Fungal infection by Colletotrichum kahawae",
    "High humidity and prolonged wet conditions",
    "Spores spread through wind, rain, and infected plant debris",
    "Weak or stressed coffee plants are more vulnerable",
    "Poor farm sanitation leading to fungal buildup",
    "Lack of resistant coffee varieties in susceptible areas"
  ],
  "remedies": {
    "organic": {
      "pruning_and_sanitation": "Remove infected berries and leaves, and prune trees to improve air circulation",
      "shade_management": "Adjust shade levels to reduce excessive moisture retention",
      "biological_control": {
        "Trichoderma": "Apply Trichoderma spp. to suppress Colletotrichum kahawae",
        "Bacillus_based_control": "Use Bacillus subtilis biofungicides to inhibit fungal growth"
      },
      "botanical_sprays": [
        "Neem oil spray to prevent fungal spore development",
        "Garlic and chili-based extracts as natural antifungal treatments"
      ],
      "organic_fertilization": "Apply compost and potassium-rich fertilizers to strengthen plant immunity"
    },
    "inorganic": {
      "copper_based_fungicides": [
        "Bordeaux mixture (copper sulfate + lime)",
        "Copper oxychloride",
        "Copper hydroxide"
      ],
      "systemic_fungicides": [
        "Propiconazole",
        "Difenoconazole"
      ],
      "contact_fungicides": [
        "Mancozeb",
        "Chlorothalonil"
      ],
      "preventive_spraying": "Apply fungicides before peak rainy seasons to minimize infection risk"
    }
  },
  "best_practices": [
    "Regularly monitor coffee farms for early symptoms of CBD",
    "Combine biological and chemical control methods for effective disease management",
    "Use resistant coffee varieties like Ruiru 11 and Batian",
    "Maintain proper farm sanitation, pruning, and nutrition to prevent fungal spread"
  ]
}

coffee_miner={
  "disease": "Coffee Leaf Miner Disease",
  "causes": [
    "Infestation by Leucoptera coffeella larvae",
    "Adult moths lay eggs on coffee leaves, and larvae tunnel through leaf tissue",
    "Hot and dry weather conditions favor outbreaks",
    "Poor farm management, including overgrown trees and lack of pruning",
    "Absence of natural predators due to excessive pesticide use",
    "Lack of regular monitoring, leading to undetected infestations"
  ],
  "remedies": {
    "organic": {
      "pruning_and_sanitation": "Regularly prune coffee trees to improve air circulation and remove infested leaves",
      "biological_control": {
        "parasitic_wasps": "Release natural enemies like Stenomalus sp. and Mirax insularis to control larvae",
        "predatory_insects": "Encourage beneficial insects like ladybugs and lacewings"
      },
      "botanical_sprays": [
        "Neem oil to disrupt leaf miner development",
        "Garlic or chili-based extracts as natural repellents"
      ],
      "handpicking_and_trapping": [
        "Physically remove and destroy infested leaves",
        "Use sticky traps to monitor and capture adult moths"
      ],
      "shade_management": "Maintain optimal shade levels to create unfavorable conditions for leaf miners"
    },
    "inorganic": {
      "selective_insecticides": [
        "Bacillus thuringiensis (Bt) as a microbial insecticide",
        "Spinosad-based insecticides that target larvae while being safe for beneficial insects"
      ],
      "systemic_insecticides": [
        "Imidacloprid to control larvae and adult leaf miners",
        "Thiamethoxam for long-term pest suppression"
      ],
      "preventive_spraying": "Apply insecticides before dry seasons when leaf miners are most active"
    }
  },
  "best_practices": [
    "Regularly inspect coffee plants for early signs of leaf miner infestation",
    "Integrate biological and selective chemical control methods",
    "Encourage natural predators to maintain ecological balance",
    "Implement farm hygiene by removing and destroying affected leaves"
  ]
}

coffee_rust={
  "disease": "Coffee Leaf Rust",
  "causes": [
    "Fungal infection by Hemileia vastatrix",
    "High humidity and warm temperatures (15–28°C)",
    "Spores spread through wind, rain, and contaminated tools",
    "Dense planting and poor air circulation create favorable conditions",
    "Weak or stressed plants due to nutrient deficiencies or drought",
    "Lack of resistant coffee varieties in affected regions"
  ],
  "remedies": {
    "organic": {
      "pruning_and_sanitation": "Remove infected leaves and prune trees to improve air circulation",
      "shade_management": "Adjust shading to prevent excessive humidity buildup",
      "biological_control": {
        "Trichoderma": "Apply Trichoderma spp. to suppress fungal spores",
        "Bacillus_subtilis": "Use Bacillus subtilis biofungicides to inhibit rust development"
      },
      "botanical_sprays": [
        "Neem oil spray to prevent spore germination",
        "Garlic and chili-based extracts as natural antifungal treatments"
      ],
      "organic_fertilization": "Apply compost and potassium-rich fertilizers to strengthen plant immunity"
    },
    "inorganic": {
      "copper_based_fungicides": [
        "Bordeaux mixture (copper sulfate + lime)",
        "Copper oxychloride",
        "Copper hydroxide"
      ],
      "systemic_fungicides": [
        "Propiconazole",
        "Tebuconazole"
      ],
      "contact_fungicides": [
        "Mancozeb",
        "Chlorothalonil"
      ],
      "preventive_spraying": "Apply fungicides before the rainy season to minimize infection risk"
    }
  },
  "best_practices": [
    "Regularly monitor coffee farms for early signs of leaf rust",
    "Use a combination of organic and chemical control methods",
    "Plant rust-resistant coffee varieties like Castillo, Catimor, and Ruiru 11",
    "Maintain proper pruning, shading, and fertilization to improve plant resistance"
  ]
}

cocoa_normal = {
    'disease': 'Normal Cocoa',
    'causes': ['N/A'],
    'remedies': {
        'organic': 'N/A',
        'inorganic': 'N/A'
    },
    'best_practices': ['N/A']
}

coffee_normal = {
    'disease': 'Normal Coffee',
    'causes': ['N/A'],
    'remedies': {
        'organic': 'N/A',
        'inorganic': 'N/A'
    },
    'best_practices': ['N/A']
}


diseases = [cocoa_blackpod, cocoa_frostypod, cocoa_mirid, cocoa_normal, coffee_browneye, coffee_miner, coffee_normal, coffee_rust]

class_labels=[d['disease'] for d in diseases]





# Initialize FastAPI app
app = FastAPI()

# ... (keep all the existing code until the classify_image endpoint)

@app.post("/classify")
async def classify_image(image: UploadFile = File(...)):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="No file received")

        file_content = await image.read()
        print(f"Received file size: {len(file_content)} bytes")  # Log file size

        # Try to open the image
        try:
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")    
        
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(DEVICE)
    
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
        
        try:
            class_name = class_labels[predicted_class.item()]
            disease_info = diseases[predicted_class.item()]
            
            # Extract the structured information
            response_data = {
                "class": class_name,
                "causes": disease_info.get("causes", "No causes information available"),
                "organic_remedies": disease_info.get("remedies", {}).get("organic", "No organic remedies available"),
                "inorganic_remedies": disease_info.get("remedies", {}).get("inorganic", "No inorganic remedies available"),
                "best_practices": disease_info.get("best_practices", "No best practices available")
            }
            
            return response_data
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Print full error message
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")