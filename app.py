# =========================
# IMPORTS
# =========================

from flask import Flask, render_template, request   # Framework web Flask
from pipeline.prediction_pipeline import hybrid_recommendation  # Pipeline de recomendación


# =========================
# FLASK APP INITIALIZATION
# =========================

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)
   # Crea la aplicación Flask


# =========================
# HOME ROUTE
# =========================
# Ruta principal que maneja GET y POST
# GET  -> muestra el formulario
# POST -> recibe el userID y devuelve recomendaciones

@app.route('/', methods=['GET', 'POST'])
def home():

    # Variable donde se almacenarán las recomendaciones
    recommendations = None

    # Si el formulario se envía (POST)
    if request.method == 'POST':
        try:
            # Obtiene el userID desde el formulario HTML y lo convierte a entero
            user_id = int(request.form["userID"])

            # Ejecuta el sistema híbrido de recomendación
            recommendations = hybrid_recommendation(user_id)

        except Exception as e:
            # Manejo básico de errores
            print("Erorr occured....")

    # Renderiza la plantilla HTML y pasa las recomendaciones
    return render_template(
        'index.html',
        recommendations=recommendations
    )


# =========================
# MAIN ENTRY POINT
# =========================
# Ejecuta la aplicación Flask

if __name__ == "__main__":
    app.run(
        debug=True,          # Modo debug (solo desarrollo)
        host='0.0.0.0',      # Accesible desde cualquier IP
        port=5000            # Puerto de la aplicación
    )
