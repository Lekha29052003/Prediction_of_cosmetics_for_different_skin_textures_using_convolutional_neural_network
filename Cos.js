// DOM elements
const input = document.querySelector("#upload");
const imagePreview = document.querySelector(".image-preview");
const analyzeButton = document.querySelector("#analyze-btn");
const resultsSection = document.querySelector("#results");
const productName = document.querySelector("#product-name");
const productDescription = document.querySelector("#product-description");
const buyButton = document.querySelector("#buy-btn");

// Load image preview
input.addEventListener("change", () => {
  const file = input.files[0];
  const reader = new FileReader();

  reader.addEventListener("load", () => {
    imagePreview.style.backgroundImage = `url(${reader.result})`;
  });

  if (file) {
    reader.readAsDataURL(file);
  }
});

// Analyze skin
analyzeButton.addEventListener("click", () => {
  // TODO: Implement skin analysis algorithm

  // Show results
  resultsSection.classList.remove("hidden");
});

// Recommend product
buyButton.addEventListener("click", () => {
  // TODO: Implement product recommendation algorithm

  // Update product information
  productName.textContent = "Product Name";
  productDescription.textContent = "Product Description";

  // Show product recommendation
  document.querySelector("#product-recommendation").classList.remove("hidden");
});
