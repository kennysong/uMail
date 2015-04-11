class ThreadController < ApplicationController
  before_filter :login_required
  layout 'framework'
  def index
    list
    render :action => 'list'
  end

  # GETs should be safe (see http://www.w3.org/2001/tag/doc/whenToUseGet.html)
  verify :method => :post, :only => [ :destroy, :create, :update ],
         :redirect_to => { :action => :list }

  def list
    @email_threads = EmailThread.find(:all, :order => 'listno')
  end

  def new
    @email_thread = EmailThread.new
    @emails = Email.find(:all, :order => 'Date')
  end

  #Create an email thread entry
  def create
    @email_thread = EmailThread.new(params[:email_thread])
    for email in Email.find(params[:email_ids])
      email.email_thread = @email_thread
      if !email.save
        render :action => 'new'
      end
    end
    if @email_thread.save
      flash[:notice] = 'EmailThread was successfully created.'
      redirect_to :action => 'list'
    else
      render :action => 'new'
    end
  end

  def edit
    @email_thread = EmailThread.find(params[:id])
    @emails = Email.find(:all, :order => 'Date')
  end

  def update
    @email_thread = EmailThread.find(params[:id])
    @emails = Email.find(:all, :order => 'Date')
    if !params[:email_ids]
      flash[:error] = 'You have to select at least one email.'
      render :action => "edit"
      return
    end
    #Tag selected emails
    for email in Email.find(params[:email_ids])
      email.email_thread = @email_thread
      if !email.save
        render :action => 'edit'
      end
    end
    #Remove unselected emails
    for email in @email_thread.emails
      if !params[:email_ids].include? email.id.to_s
        email.update_attribute('email_thread_id',nil)
      end
    end
    if @email_thread.update_attributes(params[:email_thread])
      flash[:notice] = 'EmailThread was successfully updated.'
      redirect_to :action => 'list'
    else
      render :action => 'edit'
    end
  end

  def destroy
    EmailThread.find(params[:id]).destroy
    redirect_to :action => 'list'
  end
  #Delete thread and the emails it contains
  def destroyAll
    mythread = EmailThread.find(params[:id])
    if mythread.summaries.empty?
      mythread.emails.each(&:destroy)
      mythread.destroy
    else
      flash[:notice] = "That thread has summaries associated with it: #{mythread.summaries[0].experiment.Location}"
    end
    redirect_to :action => 'list'
  end
  
  #export template
  def export
    @email_threads = EmailThread.find(:all, :order => 'listno')
  end
  
  #export emails and threads as xml files
  def process_export
    ids = params[:thread_ids]
    @threads = []
    ids.each do |id|
      @threads << EmailThread.find(id)
    end
    render :layout => false
  end
end
