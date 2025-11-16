"""
Organic Remedy Database for Crop Diseases
Provides natural treatment recommendations for each disease
"""

REMEDIES = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "remedies": [
            "Apply neem oil spray (2-3ml per liter osf water) every 10-14 days",
            "Use baking soda solution (1 tablespoon per gallon of water) as a preventive spray",
            "Remove and destroy infected leaves and fruit to prevent spread",
            "Apply compost tea to boost plant immunity",
            "Ensure proper air circulation by pruning overcrowded branches",
            "Use sulfur-based organic fungicides early in the season"
        ],
        "prevention": [
            "Plant disease-resistant apple varieties",
            "Maintain proper spacing between trees",
            "Clean up fallen leaves in autumn",
            "Apply organic mulch to prevent soil splash"
        ]
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "remedies": [
            "Prune infected branches 6-8 inches below visible cankers",
            "Apply copper-based organic fungicides during wet weather",
            "Use neem oil spray to control fungal growth",
            "Remove all mummified fruit from trees",
            "Apply compost tea to strengthen plant defenses",
            "Use hydrogen peroxide solution (1 part 3% H2O2 to 9 parts water)"
        ],
        "prevention": [
            "Avoid overhead watering",
            "Prune for good air circulation",
            "Remove dead wood and cankers",
            "Harvest fruit promptly when ripe"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "name": "Cedar Apple Rust",
        "remedies": [
            "Remove nearby juniper/cedar trees if possible (they host the disease)",
            "Apply sulfur-based fungicides in early spring",
            "Use neem oil spray every 7-10 days during infection period",
            "Remove and destroy infected leaves",
            "Apply compost tea to boost immunity"
        ],
        "prevention": [
            "Plant rust-resistant apple varieties",
            "Maintain distance from juniper trees (at least 500 feet)",
            "Apply preventive sprays before symptoms appear"
        ]
    },
    "Apple___healthy": {
        "name": "Healthy Apple",
        "remedies": [
            "Continue current care practices",
            "Maintain regular organic fertilization",
            "Monitor for early signs of disease",
            "Keep trees well-watered and mulched"
        ],
        "prevention": [
            "Regular inspection of leaves and fruit",
            "Maintain proper pruning schedule",
            "Apply preventive organic sprays in spring"
        ]
    },
    "Blueberry___healthy": {
        "name": "Healthy Blueberry",
        "remedies": [
            "Continue current care practices",
            "Maintain acidic soil pH (4.5-5.5)",
            "Apply organic mulch (pine needles or sawdust)",
            "Regular watering with rainwater when possible"
        ],
        "prevention": [
            "Test soil pH annually",
            "Prune old canes to encourage new growth",
            "Protect from birds with netting"
        ]
    },
    "Cherry_(including_sour)___healthy": {
        "name": "Healthy Cherry",
        "remedies": [
            "Continue current care practices",
            "Maintain proper pruning",
            "Apply organic compost in spring",
            "Monitor for pests and diseases"
        ],
        "prevention": [
            "Regular inspection",
            "Proper spacing",
            "Good air circulation"
        ]
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "name": "Cherry Powdery Mildew",
        "remedies": [
            "Apply milk spray (1 part milk to 9 parts water) weekly",
            "Use baking soda solution (1 tbsp per gallon) with horticultural oil",
            "Apply neem oil spray every 7-10 days",
            "Prune to improve air circulation",
            "Remove and destroy infected leaves",
            "Use sulfur-based organic fungicides"
        ],
        "prevention": [
            "Plant in areas with good air circulation",
            "Avoid overhead watering",
            "Prune regularly to maintain open canopy",
            "Apply preventive sprays in early spring"
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Corn Gray Leaf Spot",
        "remedies": [
            "Apply copper-based fungicides early in season",
            "Use neem oil spray to control fungal growth",
            "Rotate crops to break disease cycle",
            "Remove crop residue after harvest",
            "Apply compost tea to boost plant health",
            "Use resistant corn varieties in future plantings"
        ],
        "prevention": [
            "Practice crop rotation (3-4 year cycle)",
            "Plant disease-resistant varieties",
            "Avoid planting in same location consecutively",
            "Maintain proper plant spacing"
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "name": "Corn Common Rust",
        "remedies": [
            "Apply sulfur-based organic fungicides",
            "Use neem oil spray every 7-10 days",
            "Remove and destroy infected leaves if severe",
            "Apply compost tea to strengthen plants",
            "Ensure adequate nitrogen fertilization"
        ],
        "prevention": [
            "Plant rust-resistant corn varieties",
            "Practice crop rotation",
            "Avoid planting too early in cool, wet conditions",
            "Maintain proper plant spacing"
        ]
    },
    "Corn_(maize)___healthy": {
        "name": "Healthy Corn",
        "remedies": [
            "Continue current practices",
            "Maintain proper fertilization",
            "Monitor for pests and diseases",
            "Ensure adequate water during critical growth stages"
        ],
        "prevention": [
            "Crop rotation",
            "Proper spacing",
            "Regular monitoring"
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Corn Northern Leaf Blight",
        "remedies": [
            "Apply copper-based fungicides preventively",
            "Use neem oil spray to control spread",
            "Remove crop residue after harvest",
            "Practice crop rotation (3+ years)",
            "Apply compost tea to boost immunity",
            "Plant resistant varieties in future"
        ],
        "prevention": [
            "Use disease-resistant corn varieties",
            "Practice crop rotation",
            "Plow under crop residue",
            "Avoid overhead irrigation"
        ]
    },
    "Grape___Black_rot": {
        "name": "Grape Black Rot",
        "remedies": [
            "Apply copper-based fungicides before and after bloom",
            "Use neem oil spray every 7-10 days during growing season",
            "Remove and destroy all infected fruit and leaves",
            "Prune to improve air circulation",
            "Apply compost tea to strengthen vines",
            "Use sulfur-based fungicides in early season"
        ],
        "prevention": [
            "Plant disease-resistant grape varieties",
            "Prune for good air circulation",
            "Remove mummified fruit",
            "Apply preventive sprays before wet weather"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "name": "Grape Esca (Black Measles)",
        "remedies": [
            "Prune out infected canes and wood",
            "Apply compost tea to boost vine health",
            "Use mycorrhizal fungi to improve root health",
            "Ensure proper drainage",
            "Apply organic fertilizers to strengthen vines",
            "Remove and destroy infected plant material"
        ],
        "prevention": [
            "Avoid wounding vines during pruning",
            "Use clean pruning tools",
            "Maintain vine health through proper nutrition",
            "Plant disease-free stock"
        ]
    },
    "Grape___healthy": {
        "name": "Healthy Grape",
        "remedies": [
            "Continue current care",
            "Maintain proper pruning",
            "Monitor for diseases",
            "Apply organic fertilizers as needed"
        ],
        "prevention": [
            "Regular inspection",
            "Proper trellising",
            "Good air circulation"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Grape Leaf Blight",
        "remedies": [
            "Apply copper-based fungicides",
            "Use neem oil spray every 7-10 days",
            "Remove and destroy infected leaves",
            "Prune for better air circulation",
            "Apply compost tea",
            "Use sulfur-based organic fungicides"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Proper pruning",
            "Good air circulation",
            "Preventive sprays in spring"
        ]
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "name": "Citrus Greening (Huanglongbing)",
        "remedies": [
            "Control Asian citrus psyllid (vector) with neem oil",
            "Apply compost tea to boost tree health",
            "Use beneficial insects like ladybugs",
            "Apply organic fertilizers to strengthen trees",
            "Remove severely infected branches",
            "Use reflective mulches to deter psyllids"
        ],
        "prevention": [
            "Plant disease-free certified trees",
            "Monitor for psyllid presence",
            "Remove infected trees to prevent spread",
            "Use physical barriers or netting"
        ]
    },
    "Peach___Bacterial_spot": {
        "name": "Peach Bacterial Spot",
        "remedies": [
            "Apply copper-based bactericides early in season",
            "Use neem oil spray to reduce bacterial spread",
            "Prune to improve air circulation",
            "Remove and destroy infected leaves and fruit",
            "Apply compost tea to boost immunity",
            "Avoid overhead watering"
        ],
        "prevention": [
            "Plant disease-resistant varieties",
            "Avoid planting in low-lying, humid areas",
            "Use drip irrigation instead of overhead",
            "Apply preventive copper sprays in spring"
        ]
    },
    "Peach___healthy": {
        "name": "Healthy Peach",
        "remedies": [
            "Continue current practices",
            "Maintain proper pruning",
            "Monitor for diseases",
            "Apply organic fertilizers"
        ],
        "prevention": [
            "Regular inspection",
            "Proper spacing",
            "Good care practices"
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "name": "Bell Pepper Bacterial Spot",
        "remedies": [
            "Apply copper-based bactericides",
            "Use neem oil spray every 7-10 days",
            "Remove and destroy infected plants if severe",
            "Avoid overhead watering",
            "Apply compost tea",
            "Use resistant varieties for future plantings"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Practice crop rotation",
            "Avoid working with wet plants",
            "Use drip irrigation"
        ]
    },
    "Pepper,_bell___healthy": {
        "name": "Healthy Bell Pepper",
        "remedies": [
            "Continue current care",
            "Maintain proper watering",
            "Monitor for pests",
            "Apply organic fertilizers"
        ],
        "prevention": [
            "Regular monitoring",
            "Proper spacing",
            "Good soil health"
        ]
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "remedies": [
            "Apply copper-based fungicides preventively",
            "Use neem oil spray every 7-10 days",
            "Remove infected leaves",
            "Practice crop rotation (3-4 years)",
            "Apply compost tea",
            "Ensure adequate potassium in soil"
        ],
        "prevention": [
            "Use disease-free seed potatoes",
            "Practice crop rotation",
            "Avoid overhead watering",
            "Plant resistant varieties"
        ]
    },
    "Potato___healthy": {
        "name": "Healthy Potato",
        "remedies": [
            "Continue current practices",
            "Maintain proper hilling",
            "Monitor for diseases",
            "Ensure adequate water"
        ],
        "prevention": [
            "Crop rotation",
            "Proper spacing",
            "Regular monitoring"
        ]
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "remedies": [
            "Apply copper-based fungicides immediately",
            "Use neem oil spray every 5-7 days during wet weather",
            "Remove and destroy all infected plants",
            "Apply compost tea to remaining plants",
            "Ensure good drainage",
            "Harvest early if infection is severe"
        ],
        "prevention": [
            "Use certified disease-free seed potatoes",
            "Practice crop rotation",
            "Avoid overhead irrigation",
            "Plant resistant varieties",
            "Destroy all volunteer potatoes"
        ]
    },
    "Raspberry___healthy": {
        "name": "Healthy Raspberry",
        "remedies": [
            "Continue current care",
            "Maintain proper pruning",
            "Monitor for diseases",
            "Apply organic mulch"
        ],
        "prevention": [
            "Regular inspection",
            "Proper trellising",
            "Good air circulation"
        ]
    },
    "Soybean___healthy": {
        "name": "Healthy Soybean",
        "remedies": [
            "Continue current practices",
            "Maintain proper fertilization",
            "Monitor for pests",
            "Ensure adequate water"
        ],
        "prevention": [
            "Crop rotation",
            "Proper spacing",
            "Regular monitoring"
        ]
    },
    "Squash___Powdery_mildew": {
        "name": "Squash Powdery Mildew",
        "remedies": [
            "Apply milk spray (1:9 ratio) weekly",
            "Use baking soda solution (1 tbsp per gallon) with oil",
            "Apply neem oil spray every 7-10 days",
            "Remove severely infected leaves",
            "Improve air circulation",
            "Use sulfur-based fungicides"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Avoid overhead watering",
            "Maintain proper spacing",
            "Apply preventive sprays"
        ]
    },
    "Strawberry___healthy": {
        "name": "Healthy Strawberry",
        "remedies": [
            "Continue current care",
            "Maintain proper mulching",
            "Monitor for diseases",
            "Renovate beds annually"
        ],
        "prevention": [
            "Regular inspection",
            "Proper spacing",
            "Good air circulation"
        ]
    },
    "Strawberry___Leaf_scorch": {
        "name": "Strawberry Leaf Scorch",
        "remedies": [
            "Apply copper-based fungicides",
            "Use neem oil spray",
            "Remove infected leaves",
            "Apply compost tea",
            "Improve air circulation",
            "Use resistant varieties for replanting"
        ],
        "prevention": [
            "Plant disease-free runners",
            "Practice crop rotation",
            "Remove old leaves after harvest",
            "Maintain proper spacing"
        ]
    },
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "remedies": [
            "Apply copper-based bactericides",
            "Use neem oil spray every 7-10 days",
            "Remove severely infected plants",
            "Avoid overhead watering",
            "Apply compost tea",
            "Use resistant varieties for future plantings"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Practice crop rotation",
            "Avoid working with wet plants",
            "Sanitize tools between plants"
        ]
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "remedies": [
            "Apply copper-based fungicides preventively",
            "Use neem oil spray every 7-10 days",
            "Remove infected lower leaves",
            "Apply compost tea",
            "Ensure adequate calcium in soil",
            "Practice crop rotation"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Practice crop rotation",
            "Avoid overhead watering",
            "Stake plants for better air circulation",
            "Remove lower leaves as plant grows"
        ]
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato",
        "remedies": [
            "Continue current practices",
            "Maintain proper staking",
            "Monitor for diseases",
            "Apply organic fertilizers"
        ],
        "prevention": [
            "Regular inspection",
            "Proper spacing",
            "Good air circulation",
            "Crop rotation"
        ]
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "remedies": [
            "Apply copper-based fungicides immediately",
            "Use neem oil spray every 5-7 days",
            "Remove and destroy all infected plants immediately",
            "Apply compost tea to remaining plants",
            "Ensure good drainage",
            "Harvest green fruit if infection is severe"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Practice crop rotation",
            "Avoid overhead irrigation",
            "Plant resistant varieties",
            "Destroy volunteer tomatoes",
            "Apply preventive sprays in wet weather"
        ]
    },
    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "remedies": [
            "Improve air circulation by pruning",
            "Apply neem oil spray",
            "Remove infected leaves",
            "Use sulfur-based fungicides",
            "Reduce humidity in greenhouse",
            "Apply compost tea"
        ],
        "prevention": [
            "Maintain proper spacing",
            "Improve ventilation",
            "Avoid overhead watering",
            "Use resistant varieties"
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "remedies": [
            "Apply copper-based fungicides",
            "Use neem oil spray every 7-10 days",
            "Remove infected lower leaves",
            "Apply compost tea",
            "Practice crop rotation",
            "Ensure good air circulation"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Practice crop rotation",
            "Remove plant debris after harvest",
            "Avoid overhead watering",
            "Stake plants properly"
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "remedies": [
            "Spray with water to dislodge mites",
            "Apply neem oil spray every 5-7 days",
            "Release predatory mites (Phytoseiulus persimilis)",
            "Use insecticidal soap",
            "Apply compost tea to strengthen plants",
            "Increase humidity around plants"
        ],
        "prevention": [
            "Monitor regularly for early detection",
            "Maintain proper watering",
            "Avoid over-fertilizing with nitrogen",
            "Use reflective mulches",
            "Introduce beneficial insects"
        ]
    },
    "Tomato___Target_Spot": {
        "name": "Tomato Target Spot",
        "remedies": [
            "Apply copper-based fungicides",
            "Use neem oil spray",
            "Remove infected leaves",
            "Apply compost tea",
            "Improve air circulation",
            "Practice crop rotation"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Practice crop rotation",
            "Avoid overhead watering",
            "Maintain proper spacing"
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "remedies": [
            "Remove and destroy infected plants immediately",
            "Control aphids (vectors) with neem oil",
            "Sanitize all tools",
            "Wash hands before handling plants",
            "Use resistant varieties for replanting",
            "Apply compost tea to remaining plants"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Control aphids",
            "Sanitize tools regularly",
            "Avoid smoking near plants (tobacco mosaic)",
            "Plant resistant varieties"
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "remedies": [
            "Control whiteflies (vectors) with neem oil",
            "Use yellow sticky traps",
            "Remove and destroy infected plants",
            "Apply compost tea to remaining plants",
            "Use reflective mulches",
            "Introduce beneficial insects"
        ],
        "prevention": [
            "Use disease-free transplants",
            "Control whiteflies early",
            "Use row covers",
            "Plant resistant varieties",
            "Remove weed hosts"
        ]
    }
}


def get_remedy(disease_class):
    """Get remedy information for a disease class"""
    return REMEDIES.get(disease_class, {
        "name": disease_class.replace("_", " ").title(),
        "remedies": [
            "Apply neem oil spray as a general organic treatment",
            "Use compost tea to boost plant immunity",
            "Remove infected plant parts",
            "Ensure proper air circulation",
            "Practice crop rotation",
            "Maintain good soil health with organic matter"
        ],
        "prevention": [
            "Regular monitoring",
            "Proper spacing",
            "Good air circulation",
            "Organic soil management"
        ]
    })

