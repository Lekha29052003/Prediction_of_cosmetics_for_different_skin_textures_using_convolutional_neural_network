function validate() {
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;
    
    if (username === "" || password === "") {
      alert("Username and password are required.");
    } else if (username === "admin" && password === "password123") {
      alert("Login successful!");
    } else {
      alert("Invalid username or password.");
    }
  }
  