// Global Gmail.js object
var gmail;

// Initial check that jQuery and Gmail.js is loaded
var checkLoaded = function() {
  if(window.jQuery && window.Gmail) { uMailMain(); } 
  else { setTimeout(checkLoaded, 100); }
};

// Main function for setup
var uMailMain = function() {
    gmail = Gmail();
    gmail.observe.on("open_email", function(id) { emailOpened(id); });  
};

// Callback for when an email is opened
var emailOpened = function(id) {
  // Get email data of first email in thread
  var emailData = gmail.get.email_data(id);
  var firstEmail = emailData.threads[id]; // ID of thread == ID of first email
  var sendData = {
                    "email": firstEmail.content_plain,
                    "subject": firstEmail.subject,
                    "to": firstEmail.to,
                    "cc": firstEmail.cc
                  };

  // Send data to server
  // We do this on email load (rather than on button click) to reduce latency
  $.post("52.6.28.16/new_email", sendData)
    .done(function(d) { 
      console.log(d); 
    })
    .error(function(error) {
      console.log(error);
    })

  // Add Summarize button to email view toolbar
  addUMailButton();
};

// Add a Summarize button to the email view toolbar
// Adapted from api.tools.add_toolbar_button() in Gmail.js
var addUMailButton = function() {
  var toolbar = $("[gh='mtb']");
  while(!toolbar.hasClass('iH')) {
    // Hack to get the email view toolbar, not the main toolbar
    setTimeout(addUMailButton, 500);
    return;
  }
  while(toolbar.children().length == 1) {
    toolbar = $(toolbar).children().first();
  }

  var buttonContainer = $(document.createElement('div'));
  buttonContainer.attr('class','G-Ni J-J5-Ji');

  var button = $(document.createElement('div'));
  button.attr('class', 'T-I J-J5-Ji lS T-I-ax7 ar7');

  button.html('<div><img src="' + uMailExtensionURL + 
    'img/iconToolbar.png" style="position: relative;top: 4px;"> Summarize</div>');

  button.click(function() {
    console.log("yes");
  });

  var content = $(document.createElement('div'));
  content.attr('class','asa');

  buttonContainer.html(button);
  toolbar.append(buttonContainer);
}

checkLoaded();