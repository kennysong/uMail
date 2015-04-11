class StudyController < ApplicationController
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
    @experiments = Experiment.find(:all, :order => 'Date DESC')
    @show_part = session[:show_part]
  end

  def show
    @experiment = Experiment.find(params[:id])
  end

  def new
    @experiment = Experiment.new
    @threads = EmailThread.find(:all, :order => 'listno', :conditions => "Name != 'Example'")
  end

  #Assign email threads to the experiment
  def create
    if !params[:thread_ids]
      flash[:error] = 'You have to select at least one thread.'
      render :action => 'new'
    else
      @experiment = Experiment.new(params[:experiment])
      if @experiment.save
        flash[:notice] = 'Experiment was successfully created.'
        for thread in EmailThread.find(params[:thread_ids])
          @summary = Summary.new
          @summary.email_thread = thread
          @summary.experiment = @experiment
          @summary.save
        end
        redirect_to :action => 'list'
      else
        render :action => 'new'
      end
    end
  end

  def edit
    @experiment = Experiment.find(params[:id])
    @threads = EmailThread.find(:all, :order => 'listno', :conditions => "Name != 'Example'")
  end

  #Update what threads are in an experiment
  def update
    @experiment = Experiment.find(params[:id])
    if !params[:thread_ids]
      flash[:error] = 'You have to select at least one thread.'
      render :action => 'edit'
    else
      if @experiment.update_attributes(params[:experiment])
        
        #Create new summary if needed
        for thread in EmailThread.find(params[:thread_ids])
          if thread.summary(@experiment).nil?
            @summary = Summary.new
            @summary.email_thread = thread
            @summary.experiment = @experiment
            @summary.save
          end
        end
        #Delete old summary if needed
        @experiment.summaries.each { |s|
          if !params[:thread_ids].include? s.email_thread.id.to_s
            s.destroy
          end
        }
        flash[:notice] = 'Experiment was successfully updated.'
        redirect_to :action => 'list'
      else
        render :action => 'edit'
      end
    end
  end

  def destroy
    exp = Experiment.find(params[:id])
    exp.summaries.each(&:destroy) unless exp.summaries.empty?
    exp.participant.destroy unless exp.participant.nil?
    exp.destroy
    redirect_to :action => 'list'
  end
  
  #Show the annotation of an experiment
  def showSum
    @summary = Summary.find(params[:id])
    #Check saved data
    if @summary.Label.nil?
      flash[:error] = "No Labels saved"
      redirect_to :action => 'list'
      return
    end
    if @summary.Sent.nil?
      flash[:error] = "No Sentence Highlights saved"
      redirect_to :action => 'list'
      return
    end
    @email_thread = @summary.email_thread
    @emails = []
    email_count = 0
    for email in Email.find(:all, :order => 'Date')
      if email.email_thread == @email_thread
        email_count+=1
        s = email.Body
        open = false
        @count = 1
        while s.include? "^"
          div = "\
<img class='prop' id='prop#{email_count}.#{@count}' src='/images/prop_#{(@summary.Label.index("prop#{email_count}.#{@count}")) ? "l" : "d"}.gif'>\
<img class='req' id='req#{email_count}.#{@count}' src='/images/req_#{(@summary.Label.index("req#{email_count}.#{@count}")) ? "l" : "d"}.gif'>\
<img class='cmt' id='cmt#{email_count}.#{@count}' src='/images/cmt_#{(@summary.Label.index("cmt#{email_count}.#{@count}")) ? "l" : "d"}.gif'>\
<img class='meet' id='meet#{email_count}.#{@count}' src='/images/meet_#{(@summary.Label.index("meet#{email_count}.#{@count}")) ? "l" : "d"}.gif'>\
<img class='meta' id='meta#{email_count}.#{@count}' src='/images/meta_#{(@summary.Label.index("meta#{email_count}.#{@count}")) ? "l" : "d"}.gif'>\
<img class='subj' id='subj#{email_count}.#{@count}' src='/images/subj_#{(@summary.Label.index("subj#{email_count}.#{@count}")) ? "l" : "d"}.gif'>"
          if !open
            s[s.index("^"),1] = "#{div}<div class='clear_sentences' style=\"background-color: #{@summary.Sent.index("s#{email_count}.#{@count}") ? "yellow" : "white"};\"><b>#{email_count}.#{@count}</b>&nbsp;"
            open = true
            @count = @count + 1
          else
            s[s.index("^"),1] = '</div>'
            open = false
          end
        end
        s.gsub! "/-\\", "^"
        #Show newlines as html
        s.gsub! "\r\n", "<br>"
        s.gsub! "\n", "<br>"
        #Remove extra spaces
        s.gsub! "<br><br>", "<br>"
        email.Body = s
        @emails << email
      end
    end
  end
  
  def editSum
    @summary = Summary.find(params[:id])
  end
  
  def updateSum
    @sum = Summary.find(params[:id])
    if @sum.update_attributes(params[:summary])
      flash[:notice] = 'Summary was successfully updated.'
      redirect_to :action => 'list'
    else
      render :action => 'editSum'
    end
  end
  
  def destroySum
    Summary.find(params[:id]).destroy
    redirect_to :action => 'list'
  end
  
  #Run an experiment, so logs out administrator from the system
  def run
    @experiment = Experiment.find(params[:id])
    if @experiment.still_valid?
      @participant = @experiment.participant
      @participant ||= Participant.new
      if @participant.save
        @experiment.update_attribute('participant_id',@participant.id)
        #Store the experiment in the session
        redirect_to :controller => "account", :action => "codelogout", :path => url_for(:controller => 'summary', :action => 'userinfo'), :ses => {:id => @participant.id, :exp => @experiment.id}
      else
        flash[:error] = 'Could not create new participant'
        redirect_to :action => 'list'
      end
    else
      flash[:error] = 'This experiment cannot be run anymore'
      redirect_to :action => 'list'
    end
  end
  
  #Change the order of the threads in each experiment
  def move
    summary = Summary.find(params[:id])
    num = summary.list_order
    if (params[:direction] == 'up' && num > 0)
      candidate = Summary.find_by_list_order(num-1,:conditions => "experiment_id = #{summary.experiment.id}")
      summary.update_attribute('list_order',num-1) if !candidate.nil?
      candidate.update_attribute('list_order',num) if !candidate.nil?
    elsif (params[:direction] == 'down')
      candidate = Summary.find_by_list_order(num+1,:conditions => "experiment_id = #{summary.experiment.id}")
      summary.update_attribute('list_order',num+1) if !candidate.nil?
      candidate.update_attribute('list_order',num) if !candidate.nil?
    end
    redirect_to :action => 'list'
  end
  
  #Show or hide a participants name for privacy reasons
  def show_part
    session[:show_part] = true
    redirect_to :action => 'list'
  end
  def hide_part
    session[:show_part] = nil
    redirect_to :action => 'list'
  end
  
  #Export template
  def export
    @studies = Experiment.find(:all, :order => 'Date DESC')
  end
  
  #Export annotations as xml files
  def process_export
    ids = params[:study_ids]
    summaries = []
    #Find appropriate summaries
    ids.each do |id|
      summaries += Experiment.find(id).summaries
    end
    
    threads = {}
    summaries.each do |summary|
      if threads.has_key? summary.email_thread
        threads[summary.email_thread] << summary
      else
        threads[summary.email_thread] = [summary]
      end
    end
    @threads = threads.values
    
    render :layout => false
  end
end
