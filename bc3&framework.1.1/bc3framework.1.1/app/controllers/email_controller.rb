class EmailController < ApplicationController
  require 'tempfile'
  require 'hpricot'
  require 'cgi'
  before_filter :login_required
  layout 'framework'
  def index
    list
    render :action => 'list'
  end

  # GETs should be safe (see http://www.w3.org/2001/tag/doc/whenToUseGet.html)
  verify :method => :post, :only => [ :destroy, :create, :update ],
         :redirect_to => { :action => :list }

  #List users how submitted emails
  def list
    @users = {}
    Email.find(:all).each {|email|
      if email.Submit_By
        if @users[email.Submit_By]
          @users[email.Submit_By] += 1
        else
          @users[email.Submit_By] = 1
        end
      end
      }
  end

  #Show all emails submitted by a specific user
  def listby
    @emails = Email.find(:all)
    @by = params[:id]
  end

  #Template for new email
  def new
    @email = Email.new
  end
  
  #Create an email database entry
  def create
    @email = Email.new(params[:email])
    @email.Body = segment(@email.Body)
    if @email.save
      flash[:notice] = 'Email was successfully created.'
      redirect_to :action => 'list'
    else
      render :action => 'new'
    end
  end

  #Create an entire email thread from the passed parameters
  def create_many
    error = false
    #Create Email thread as well
    @email_thread = EmailThread.new
    @email_thread.Name = params[:e][0][:Subject]
    @email_thread.listno = params[:doc_id]
    params[:e].each do |email|
      @email = Email.new(email)
      @email.email_thread = @email_thread
      if !@email.save
        render :action => 'new'
        error = true
      end
    end
    flash[:notice] = 'Email was successfully created.' unless error
    redirect_to :action => 'list' unless error
  end
  
  #Parse an email header for the different fields
  def parse
    @email = Email.new(params[:email])
    @temp = params[:temp][:t]
    @email.Submit_By = params[:temp][:by]
    #body starts after a blank line in the header
    isbody = false
    @email.Body = ""
    @email.Original = @temp
    @temp.each { |line|
      if isbody 
        @email.Body = @email.Body + line
      else
        isbody = (line.chomp!.length == 0)
      end
    }
    
    #To
    pat = @temp[/^ *[Tt]o:[^:]*?(:|(\r\n|\n)(\r\n|\n))/m]
    @email.To = pat.sub(/ *[Tt]o: ?/,'').sub(/(\r\n|\n)[^\n\r]*\Z/,'').gsub(/(\r\n|\n)/,'') if pat
    #From
    pat = @temp[/^ *[Ff]rom:[^:]*:/m]
    @email.From = pat.sub(/ *[Ff]rom: ?/,'').sub(/(\r\n|\n)[^\n\r]*\Z/,'').gsub(/(\r\n|\n)/,'') if pat
    #Cc
    pat = @temp[/^ *[Cc][Cc]:[^:]*:/m]
    @email.Cc = pat.sub(/ *[Cc][Cc]: ?/,'').sub(/(\r\n|\n)[^\n\r]*\Z/,'').gsub(/(\r\n|\n)/,'') if pat
    #Date
    pat = @temp[/^ *(Date:|Sent:).*/]
    @email.Date = pat.sub(/ *(Date:|Sent:) ?/,'') if pat
    #Subject
    pat = @temp[/^ *Subject:.*/]
    @email.Subject = pat.sub(/ *Subject: ?/,'') if pat
  end
  
  def edit
    @email = Email.find(params[:id])
  end

  #Update an email database entry
  def update
    @email = Email.find(params[:id])
    if @email.update_attributes(params[:email])
      flash[:notice] = 'Email was successfully updated.'
      redirect_to :action => 'list'
    else
      render :action => 'edit'
    end
  end

  def destroy
    Email.find(params[:id]).destroy
    redirect_to :action => 'list'
  end
  
  def newxml
    @email = Email.new
  end
  
  #Parse an email from the W3C xml format
  def parsexml
    @filename = params[:email][:email_file].original_filename
    orig = params[:email][:email_file].read
    @emails = []
    doc = Hpricot::XML(orig)
    (doc/:DOC).each do |status| 
        email = Email.new
        email.To = status.at("TO").innerHTML.strip! if status.at("TO")
        email.Cc = status.at("CC").innerHTML.strip! if status.at("CC")
        email.Date = status.at("RECEIVED").innerHTML.strip! if status.at("RECEIVED")
        email.From = status.at("NAME").innerHTML.strip! if status.at("NAME") 
        email.From += " <"+status.at("EMAIL").innerHTML.strip!+">" if status.at("EMAIL") && email.From
        email.Subject = status.at("SUBJECT").innerHTML.strip! if status.at("SUBJECT")
        email.Body = status.at("TEXT").innerHTML.strip! if status.at("TEXT")
        email.Original = orig
        email.Submit_By = current_user.login
        email.Body = label(CGI.escapeHTML(email.Body)) if status.at("TEXT")
        id ||= status.at("DOCNO").innerHTML.strip! if status.at("DOCNO")
        @docid  ||= id[6..id.length] if id
        @emails << email
      end
  end
  
  #Automatically segment sentences
  def segment(text)
    #Use temp file to segment sentences and then label them in body
    orig_fh = Tempfile.new('body')
    seg_fh = Tempfile.new('seg')
    #Free up the '^' for marking purposes
    orig_fh.puts text.gsub('^', '/-\\')
    orig_fh.flush
    system("public/sentence-boundary.pl -d public/HONORIFICS -i #{orig_fh.path} -o #{seg_fh.path}")
    seg_fh.rewind
    # ^ is the stored marking for the beginning and end of a sentence
    label(text)
  end
  
  
  #Label the beginning and end of sentences
  def label(text)
    text.gsub!('^', '/-\\')
    arg = text.split(/\r*\n/) #split on newline
    text = arg.reject{|item| item == ""}
    text = text.join("^\n^")
    text = "^"+text+"^"
    return text
  end
end
